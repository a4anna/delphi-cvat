from sklearn.ensemble import VotingClassifier


class PretrainedVotingClassifier(VotingClassifier):

    def __init__(self, estimators):
        super().__init__(estimators=estimators, voting='soft')
        self.estimators_ = estimators
