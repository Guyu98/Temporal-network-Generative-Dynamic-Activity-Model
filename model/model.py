import numpy as np



class User(object):
    def __init__(self, user_id, activity_potential, tie_inc):
        self.user_id = user_id
        self.activity_potential = activity_potential
        self.neighborhood = Neighborhood(tie_inc)
        self.last_active = -1


class Neighborhood(object):
    def __init__(self, tie_inc):
        self.tie_inc = tie_inc
        self.weights = dict()

    def new_tie(self, user):
        assert user not in self.weights
        self.weights[user] = 1.0

    def reinforce_tie(self, user):
        assert user in self.weights
        self.weights[user] += self.tie_inc

    def delete_tie(self, user):
        assert user in self.weights
        del self.weights[user]

    def get_random_neighbor_id(self, exceptions=None):
        if exceptions is None:
            exceptions = set()

        users = []
        weights = []
        for user_id, weight in self.weights.items():
            if user_id in exceptions:
                continue
            users.append(user_id)
            weights.append(weight)

        total_weight = np.sum(weights)
        return np.random.choice(
            users, p=[weight / total_weight for weight in weights]
        )

    def get_neighbor_ids(self):
        return list(self.weights.keys())

    def __len__(self):
        return len(self.weights)


