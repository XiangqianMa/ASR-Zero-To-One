import numpy as np


class HMM:
    def __init__(self, num_states, pi, trans_prob, emit_prob):
        """
        :param pi: 初始概率矩阵
        :param trans_prob: 状态转移概率矩阵
        :param emit_prob: 发射概率矩阵
        """
        self.num_states = num_states
        self.pi = pi
        self.trans_prob = trans_prob
        self.emit_prob = emit_prob

    def forward(self, o):
        """
        前向算法
        :param o:
        :return:
        """
        time_steps = o.shape[0]
        # 初始化前向概率矩阵
        alpha = np.zeros((self.num_states, time_steps))
        for i in range(0, self.num_states):
            alpha[i][0] = self.pi[i] * self.emit_prob[i][o[0]]

        for cur_time_step in range(1, time_steps):
            for cur_state_idx in range(0, self.num_states):
                last_time_step_sum = 0
                for last_state_idx in range(0, self.num_states):
                    last_time_step_sum += alpha[last_state_idx][cur_time_step - 1] * \
                                          self.trans_prob[last_state_idx][cur_state_idx]
                alpha[cur_state_idx][cur_time_step] = last_time_step_sum * \
                                                      self.emit_prob[cur_state_idx][o[cur_time_step]]

        # 观察序列概率
        prob = 0
        for state_idx in range(0, self.num_states):
            prob += alpha[state_idx][time_steps - 1]

        return prob

    def backward(self, o):
        """
        后向算法
        :param o:
        :return:
        """
        time_steps = o.shape[0]
        # 初始化后向传播矩阵
        beta = np.zeros((self.num_states, time_steps))
        beta[:, time_steps - 1] = 1

        for cur_time_step in range(time_steps - 2, -1, -1):
            for cur_state_idx in range(0, self.num_states):
                next_state_prob_sum = 0
                for next_state_idx in range(0, self.num_states):
                    next_state_prob_sum += self.trans_prob[cur_state_idx][next_state_idx] * \
                                           self.emit_prob[next_state_idx][o[cur_time_step + 1]] * \
                                           beta[next_state_idx][cur_time_step + 1]
                beta[cur_state_idx][cur_time_step] = next_state_prob_sum

        prob = 0
        for i in range(0, self.num_states):
            prob += self.pi[i] * self.emit_prob[i][o[0]] * beta[i][0]

        return prob

    def viterbi(self, o):
        """
        Viterbi解码算法
        :param o:
        :return:
        """
        time_steps = o.shape[0]
        # 初始化Viterbi矩阵和路径矩阵
        delta = np.zeros((self.num_states, time_steps))
        phi = np.zeros((self.num_states, time_steps), dtype=int)
        for i in range(0, self.num_states):
            delta[i][0] = self.pi[i] * self.emit_prob[i][o[0]]
            phi[i][0] = -1

        for cur_time_step in range(1, time_steps):
            for cur_state_idx in range(0, self.num_states):
                # 获取最大转移路径
                max_last_prob = 0
                for last_state_idx in range(0, self.num_states):
                    tmp_prob = delta[last_state_idx][cur_time_step - 1] * self.trans_prob[last_state_idx][cur_state_idx]
                    if tmp_prob > max_last_prob:
                        max_last_prob = tmp_prob
                        phi[cur_state_idx][cur_time_step] = last_state_idx

                delta[cur_state_idx][cur_time_step] = max_last_prob * self.emit_prob[cur_state_idx][o[cur_time_step]]

        max_state_idx = 0
        for i in range(0, self.num_states):
            if delta[i][time_steps - 1] > delta[max_state_idx][time_steps - 1]:
                max_state_idx = i
        best_prob = delta[max_state_idx][time_steps - 1]

        best_path = [max_state_idx]
        last_state_idx = max_state_idx
        for cur_time_step in range(time_steps - 2, -1, -1):
            # 由下一个时刻获取当前状态
            cur_state_idx = phi[last_state_idx][cur_time_step + 1]
            best_path.append(cur_state_idx)
            last_state_idx = cur_state_idx

        best_path = best_path[::-1]
        return best_prob, best_path


if __name__ == '__main__':
    pi = np.asarray([0.2, 0.4, 0.4])
    trans_prob = np.asarray(
        [[0.5, 0.2, 0.3],
         [0.3, 0.5, 0.2],
         [0.2, 0.3, 0.5]]
    )
    emit_prob = np.asarray(
        [[0.5, 0.5],
         [0.4, 0.6],
         [0.7, 0.3]]
    )

    observations = np.asarray([0, 1, 0])

    hmm_model = HMM(3, pi, trans_prob, emit_prob)
    o_prob_forward = hmm_model.forward(observations)
    print("Forward Prob: ", o_prob_forward)
    o_prob_backward = hmm_model.backward(observations)
    print("Backward Prob: ", o_prob_backward)
    best_prob, best_path = hmm_model.viterbi(observations)
    # 从0开始 --> 从1开始
    for idx, value in enumerate(best_path):
        best_path[idx] = value + 1
    print("Best Prob: ", best_prob)
    print("Best Path: ", best_path)
    pass
