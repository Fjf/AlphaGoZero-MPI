"""
Monte Carlo Tree Search (MCTS)

This class will do neural-network augmented MCTS simulations, providing an improvement over the policy vector.
In the current state, the improved policy vector generates all available moves from the current board-state.
For every board it will:
 - Check if it is a terminal state
   - true:  set child_value to 0 for a loss, or 1 for a win.
   - false: retrieve the (child_policy, child_value) vector from the neural network
 - Compute policy vector index i
 - Set policy[i] to the newly retrieved child_value.
"""
import math

import numpy as np
import torch

from games.Game import Game, GameNode


class MCTS:
    def __init__(self, iterations=100, device="cpu", mcts_batch_size=16, exploration_factor=0.6):
        self.iterations = iterations
        self.device = torch.device(device)
        self.batch_size = mcts_batch_size
        self.exploration_factor = exploration_factor

    def select_best_leaf(self, root: GameNode) -> GameNode:
        parent = root

        # While we are not done, and we have expanded the node, search deeper
        while parent.winner() is None and len(parent.children) > 0:
            children = parent.children
            best_value, best = -float("inf"), None
            for i, child in enumerate(children):
                value = self.ucb(child, parent)
                if best_value < value:
                    best_value, best = value, child

            assert best is not None
            parent = best
        return parent

    def ucb(self, child, parent):
        if child.mcts.n_sims > 0:
            value = (
                    (child.mcts.value / child.mcts.n_sims) +
                    self.exploration_factor *
                    child.mcts.prior *
                    math.sqrt(parent.mcts.n_sims) /
                    (1 + child.mcts.n_sims)
            )
        else:
            value = (
                    self.exploration_factor *
                    child.mcts.prior *
                    math.sqrt(parent.mcts.n_sims + 1e-5)
            )

        return value

    @staticmethod
    def backpropagate(node: GameNode, value: float):
        leaf_to_play = node.to_play

        while node is not None:
            # Scale the values down to probabilities in the range (0, 1) instead of (-1, 1)
            if node.mcts.value is None:
                node.mcts.value = 0

            # The original value was calculated from the perspective of the leaf node
            # Any node which is not the leaf node has an inverse reward.
            node.mcts.value += value if node.to_play == leaf_to_play else -value
            node.mcts.n_sims += 1

            node = node.parent

    @staticmethod
    def expand_and_get_value(network, leaf: GameNode):
        leaf.expand()

        # Predict policy and value
        policy, value = network.predict(leaf.to_np(leaf.to_play))

        # Only valid moves should be non-zero
        valid_indices = torch.IntTensor([child.encode() for child in leaf.children])
        np_policy = np.zeros(policy.shape)
        np_policy[valid_indices] = torch.index_select(policy, 0, valid_indices).detach()

        leaf.mcts.policy = np_policy / np.sum(np_policy)

        # Set child prior
        for child in leaf.children:
            child.mcts.prior = leaf.mcts.policy[child.encode()].item()

        return value.item()

    @staticmethod
    def get_terminal_value(node: GameNode):
        winner = node.winner()
        if winner is None:
            return None

        if node.winner() == node.to_play:
            return 1.
        elif node.winner() == -1:
            return 0.
        else:
            return -1.

    def process(self, game: Game, network, temperature=1.):
        """
        Processes a node with a policy and returns the MCTS-augmented policy.
        :param temperature:
        :param network:
        :param game:
        :return:
        """
        game.reset_children()
        self.expand_and_get_value(network, game.node)
        game.node.mcts.value = 0

        # Do N iterations of MCTS, building the tree based on the NN output.
        for i in range(self.iterations):
            # Select
            leaf = self.select_best_leaf(game.node)

            # Get value if terminal
            value = self.get_terminal_value(leaf)
            if value is None:
                # Expand, then get value
                value = self.expand_and_get_value(network, leaf)

            # Backpropagate value
            self.backpropagate(leaf, value)
            # print(value, id(leaf), id(game.node))
            # print(",".join(str(child.mcts.value) for child in game.children()))

        # Create a new policy to fill with MCTS updated values
        a = self.create_policy_vector(game, temperature) + 1e-7
        # print(a)
        return a

    def create_policy_vector(self, game, temperature):
        counts = torch.zeros(game.node.mcts.policy.shape)
        for child in game.children():
            counts[child.encode()] += child.mcts.n_sims

        if temperature == 0:
            best_counts = torch.argwhere(counts == torch.max(counts)).flatten()
            best = np.random.choice(best_counts)
            policy_vector = torch.zeros(game.node.mcts.policy.shape)
            policy_vector[best] = 1
            return policy_vector

        counts = counts ** (1 / temperature)
        policy_vector = counts / torch.sum(counts)
        return policy_vector
