'''
Author: hanyu
Date: 2021-01-04 08:25:44
LastEditTime: 2021-01-04 12:40:21
LastEditors: hanyu
Description: test Env class
FilePath: /test_ppo/test/test_env.py
'''
import random

import gym_super_mario_bros
import numpy as np
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from PIL import Image
from utils.get_gaes import get_gaes
from examples.PPO_super_mario_bros.env import padding, Seg


class Env(object):
    """
    Raw single environment of game
    """

    def __init__(self, act_space, act_repeats, frames, state_size, burn_in, seqlen, game):
        '''
        description: init basic params settings.
        param {
            act_space: agent act spaces.
            act_repeats: one a repeats number, default as 1.
            frames: the number of frames of s stack.
            state_size: state_size calculated in build_policy_evaluator().
            burn_in: sequences length of each burn-in segment.
            seqlen: sequences length of each training segment.
            game: game environment.
        }
        return {None}
        '''
        self.act_space = act_space
        self.act_repeats = act_repeats
        self.act_repeat = random.choice(self.act_repeats)
        self.frames = frames
        self.state_size = state_size
        self.game = game
        self.burn_in = burn_in
        self.seqlen = seqlen

        self.max_pos = -10000

        self.count = 0

        # make gym env from gym_super_mario_bros
        env = gym_super_mario_bros.make(game)
        # warp the raw env through JoypadSpace according act_space
        if self.act_space == 7:
            self.env = JoypadSpace(env, SIMPLE_MOVEMENT)
        elif self.act_space == 12:
            self.env = JoypadSpace(env, COMPLEX_MOVEMENT)

        # resize the output image to 84*84 & normalize the pixel
        # output: (84, 84, 1)
        s_t = self.resize_image(self.env.reset())
        # expand the state dimension
        # output: (84, 84, frames)
        self.s_t = np.tile(s_t, [1, 1, frames])
        # add the batch_size dimension
        # output: (batch_size, 84, 84, frames)
        self.s = [self.s_t]

        # action shape: (batch_size, )
        self.a_t = random.randint(0, act_space - 1)
        self.a = [self.a_t]
        # action logits shape: (batch_size, act_space)
        self.a_logits = []
        self.r = [0]
        self.pos = []

        self.v_cur = []

        # decides according to build_policy_evaluator()
        state_in = np.zeros(self.state_size, dtype=np.float32)
        # state_in shape: (batch_size, state_in_number)
        self.state_in = [state_in]

        self.done = False

    def step(self, a, a_logits, v_cur, state_in, force=False):
        '''
        description: step function
        param {
            a: step action
            a_logits: action logits
            v_cur: current value
            state_in: state_in
            force: force flag
        }
        return {
            segs: list of ["s", "a", "a_logits", "r", "gaes", "v_cur", "state_in"]
        }
        '''
        # repeat the last action or step the current action
        # according to the act_repeat
        self.count += 1
        if self.count % self.act_repeat == 0:
            self.a_t = a
            self.count = 0
            self.act_repeat = random.choice(self.act_repeats)

        # step the action and get the result
        gs_t1, gr_t, gdone, ginfo = self.env.step(self.a_t)
        if not gdone:
            s_t1, r_t, done, info = self.env.step(self.a_t)
            r_t += gr_t
            r_t /= 2.
        else:
            s_t1 = gs_t1
            r_t = gr_t
            done = gdone
            info = ginfo
        # reward scaling
        r_t /= 15.
        s_t1 = self.resize_image(s_t1)
        channels = s_t1.shape[-1]
        # concatenate s_t1(the last stacked frame) to self.s_t(drop the stacked first frame)
        self.s_t = np.concatenate(
            [s_t1, self.s_t[:, :, :-channels]], axis=-1)

        self.s.append(self.s_t)
        self.a.append(self.a_t)
        self.a_logits.append(a_logits)
        self.r.append(r_t)
        self.max_pos = max(self.max_pos, info["x_pos"])
        self.pos.append(info["x_pos"])
        if (len(self.pos) > 100) and (
                info["x_pos"] - self.pos[-100] < 5) and (
                self.pos[-100] - info["x_pos"] < 5):
            done = True
        self.done = done

        self.v_cur.append(v_cur)
        self.state_in.append(state_in)

        """
            get segs
            """
        segs = self.get_history(force)

        """
            reset env
            """
        self.reset(force)

        return segs

    def reset(self, force=False):
        if self.done or force:
            max_pos = self.max_pos
            self.max_pos = -10000
            print("  Max Position  %s : %d" % (self.game, max_pos))
            self.count = 0
            self.act_repeat = random.choice(self.act_repeats)

            s_t = self.resize_image(self.env.reset())

            self.s_t = np.tile(s_t, [1, 1, self.frames])
            self.s = [self.s_t]

            self.a_t = random.randint(0, self.act_space - 1)
            self.a = [self.a_t]
            self.a_logits = []
            self.r = [0]
            self.pos = []

            self.v_cur = []

            state_in = np.zeros(self.state_size, dtype=np.float32)
            self.state_in = [state_in]

            self.done = False

    def get_state(self):
        return self.s_t

    def get_act(self):
        return self.a_t

    def get_max_pos(self):
        return self.max_pos

    def reset_max_pos(self):
        self.max_pos = -10000

    def get_state_in(self):
        return self.state_in[-1]

    def get_history(self, force=False):
        if self.done or force:
            if self.done:
                gaes, _ = get_gaes.get_gaes(None, self.r, self.v_cur,
                                            self.v_cur[1:] + [0], 0.99, 0.95)
                seg = Seg(self.s, self.a, self.a_logits, self.r,
                          gaes, self.v_cur, self.state_in)
                return self.postprocess(seg)
            if force and len(self.r) > 1:
                gaes = get_gaes.get_gaes(
                    None, self.r[:-1], self.v_cur[:-1], self.v_cur[1:], 0.99, 0.95)[0]
                seg = Seg(self.s[:-1], self.a[:-1], self.a_logits[:-1], self.r[:-1], gaes,
                          self.v_cur[:-1], self.state_in[:-1])
                return self.postprocess(seg)
        return None

    @staticmethod
    def resize_image(image, size=84):
        image = Image.fromarray(image)
        image = image.convert("L")
        image = image.resize((size, size))
        image = np.array(image)
        image = image / 255.
        image = np.array(image, np.float32)
        return image[:, :, None]

    def postprocess(self, seg):
        """
        postprocess the seg for training
        """
        burn_in = self.burn_in
        seqlen = self.seqlen + burn_in
        seg_results = []
        if seg is not None:
            while len(seg[0]) > burn_in:
                next_seg = dict()
                # padding the segment
                next_seg["s"] = padding(seg.s[:seqlen], seqlen, np.float32)
                next_seg["a"] = padding(
                    seg.a[1:seqlen + 1], seqlen, np.int32)
                next_seg["prev_a"] = padding(
                    seg.a[:seqlen], seqlen, np.int32)
                next_seg["a_logits"] = padding(
                    seg.a_logits[:seqlen], seqlen, np.float32)
                next_seg["r"] = padding(
                    seg.r[1:seqlen + 1], seqlen, np.float32)
                next_seg["prev_r"] = padding(
                    seg.r[:seqlen], seqlen, np.float32)
                next_seg["adv"] = padding(
                    seg.gaes[:seqlen], seqlen, np.float32)
                next_seg["v_cur"] = padding(
                    seg.v_cur[:seqlen], seqlen, np.float32)
                next_seg["state_in"] = np.array(
                    seg.state_in[0], np.float32)
                next_seg["slots"] = padding(
                    len(seg.s[:seqlen]) * [1], seqlen, np.int32)

                seg_results.append(next_seg)
                seg = Seg(*[t[burn_in:] for t in seg])
        if any(seg_results):
            # print("full use one segs done!")
            return seg_results
        else:
            return None


def test_main():
    test_env = Env(act_space=12,
                   act_repeats=[1],
                   frames=1,
                   state_size=256*2,
                   seqlen=32,
                   burn_in=32,
                   game="SuperMarioBros-1-2-v0")

    test_env.reset()
    while not test_env.done:
        a = random.randint(0, 11)
        test_env.step(a, 1, 2, np.zeros((1, 256*2)))


if __name__ == "__main__":
    test_main()
