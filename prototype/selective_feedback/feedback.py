"""Only same-context observations are evidence; missing is not failure."""
from collections import defaultdict


def validate_feedback(context, draft, target, observations, accepted, committed_count):
    rows = []
    for depth, predictions in observations.items():
        if len(predictions) != len(target):
            raise ValueError('Observer/target alignment differs')
        for position, (prediction, truth) in enumerate(zip(predictions, target)):
            valid = position <= accepted and position < committed_count
            rows.append(dict(depth=depth, position=position,
                             prefix_ids=context + draft[:position],
                             prediction=prediction, target=truth,
                             valid=valid, match=(prediction == truth) if valid else None,
                             reason='committed_context' if valid else 'rejected_or_after_stop',
                             evidence='one_step_only', counterfactual_energy_j=None))
    return rows


class LearningState:
    """Descriptive one-step match statistics, NOT a rollout or energy estimator."""
    def __init__(self):
        self.counts = defaultdict(lambda: [0, 0])

    def update(self, rows):
        for row in rows:
            if row['valid']:
                count = self.counts[row['depth']]
                count[0] += int(row['match'])
                count[1] += 1

    def export(self):
        return {str(d): dict(matches=m, observations=n, rate=m/n)
                for d, (m, n) in self.counts.items() if n}
