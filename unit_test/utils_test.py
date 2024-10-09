import unittest

import torch.nn as nn

from neural_dnf.neural_dnf import NeuralDNF, NeuralDNFEO, NeuralDNFFullMutexTanh
from neural_dnf.utils import DeltaDelayedExponentialDecayScheduler


class TestDeltaDelayedExponentialDecayScheduler(unittest.TestCase):
    def test_decay_scheduler_only_takes_in_certain_modules(self):
        DeltaDelayedExponentialDecayScheduler(
            initial_delta=0.1,
            delta_decay_delay=5,
            delta_decay_steps=3,
            delta_decay_rate=1.1,
            target_module_type=NeuralDNF.__name__,
        )

        DeltaDelayedExponentialDecayScheduler(
            initial_delta=0.1,
            delta_decay_delay=5,
            delta_decay_steps=3,
            delta_decay_rate=1.1,
            target_module_type=NeuralDNFEO.__name__,
        )

        DeltaDelayedExponentialDecayScheduler(
            initial_delta=0.1,
            delta_decay_delay=5,
            delta_decay_steps=3,
            delta_decay_rate=1.1,
            target_module_type=NeuralDNFFullMutexTanh.__name__,
        )

        model = nn.Sequential(
            nn.Linear(3, 5),
            nn.ReLU(),
            nn.Linear(5, 4),
        )
        with self.assertRaises(AssertionError):
            DeltaDelayedExponentialDecayScheduler(
                initial_delta=0.1,
                delta_decay_delay=5,
                delta_decay_steps=3,
                delta_decay_rate=1.1,
                target_module_type=model.__class__.__name__,
            )

    def test_decay_scheduler_checks_the_module_type_before_step(self):
        model = NeuralDNF(3, 5, 4, 0.1)
        ded_scheduler = DeltaDelayedExponentialDecayScheduler(
            initial_delta=0.1,
            delta_decay_delay=5,
            delta_decay_steps=3,
            delta_decay_rate=1.1,
            target_module_type=model.__class__.__name__,
        )

        with self.assertRaises(AssertionError):
            ded_scheduler.step(model.conjunctions)

    def test_step_calculates_the_correct_new_delta_for_ndnf_model(self):
        INITIAL_DELTA = 0.1
        DECAY_RATE = 1.1

        # Create a neural DNF module
        model = NeuralDNF(3, 5, 4, INITIAL_DELTA)

        # Create a scheduler that decays delta by a factor of 0.1 every 3 steps,
        # delayed by 5 steps
        ded_scheduler = DeltaDelayedExponentialDecayScheduler(
            initial_delta=INITIAL_DELTA,
            delta_decay_delay=5,
            delta_decay_steps=3,
            delta_decay_rate=DECAY_RATE,
            target_module_type=model.__class__.__name__,
        )

        # Expectations for the first 20 steps
        expectations = [
            INITIAL_DELTA,
            INITIAL_DELTA,
            INITIAL_DELTA,
            INITIAL_DELTA,
            INITIAL_DELTA,
            INITIAL_DELTA * DECAY_RATE,
            INITIAL_DELTA * DECAY_RATE,
            INITIAL_DELTA * DECAY_RATE,
            INITIAL_DELTA * (DECAY_RATE**2),
            INITIAL_DELTA * (DECAY_RATE**2),
            INITIAL_DELTA * (DECAY_RATE**2),
            INITIAL_DELTA * (DECAY_RATE**3),
            INITIAL_DELTA * (DECAY_RATE**3),
            INITIAL_DELTA * (DECAY_RATE**3),
            INITIAL_DELTA * (DECAY_RATE**4),
            INITIAL_DELTA * (DECAY_RATE**4),
            INITIAL_DELTA * (DECAY_RATE**4),
            INITIAL_DELTA * (DECAY_RATE**5),
            INITIAL_DELTA * (DECAY_RATE**5),
            INITIAL_DELTA * (DECAY_RATE**5),
        ]

        for i in range(20):
            ret_dict = ded_scheduler.step(model)

            new_delta_vals = ret_dict["new_delta_vals"]
            old_delta_vals = ret_dict["old_delta_vals"]
            self.assertEqual(len(new_delta_vals), 2)
            self.assertEqual(len(old_delta_vals), 2)

            expected_new_delta = expectations[i]
            expected_old_delta = expectations[i - 1] if i > 0 else INITIAL_DELTA
            for new_delta, old_delta in zip(new_delta_vals, old_delta_vals):
                self.assertEqual(new_delta, expected_new_delta)
                self.assertEqual(old_delta, expected_old_delta)

        # INITIAL_DELTA * (DECAY_RATE ** 10) should be achieved after 32 steps
        # during i = 32 to 34
        for i in range(20, 35):
            ret_dict = ded_scheduler.step(model)
            new_delta_vals = ret_dict["new_delta_vals"]
            old_delta_vals = ret_dict["old_delta_vals"]
            self.assertEqual(len(new_delta_vals), 2)
            self.assertEqual(len(old_delta_vals), 2)

            if i == 32:
                expected_new_delta = INITIAL_DELTA * (DECAY_RATE**10)
                expected_old_delta = INITIAL_DELTA * (DECAY_RATE**9)
                for new_delta, old_delta in zip(new_delta_vals, old_delta_vals):
                    self.assertEqual(new_delta, expected_new_delta)
                    self.assertEqual(old_delta, expected_old_delta)

            if i in [33, 34]:
                expected_new_delta = INITIAL_DELTA * (DECAY_RATE**10)
                expected_old_delta = INITIAL_DELTA * (DECAY_RATE**10)
                for new_delta, old_delta in zip(new_delta_vals, old_delta_vals):
                    self.assertEqual(new_delta, expected_new_delta)
                    self.assertEqual(old_delta, expected_old_delta)

        # INITIAL_DELTA * (DECAY_RATE ** 24) should be the last delta value
        # below 1.0, and should be achieved after 74 steps during i = 74 to 76
        for i in range(35, 77):
            ret_dict = ded_scheduler.step(model)
            new_delta_vals = ret_dict["new_delta_vals"]
            old_delta_vals = ret_dict["old_delta_vals"]
            self.assertEqual(len(new_delta_vals), 2)
            self.assertEqual(len(old_delta_vals), 2)

            if i == 74:
                expected_new_delta = INITIAL_DELTA * (DECAY_RATE**24)
                expected_old_delta = INITIAL_DELTA * (DECAY_RATE**23)
                for new_delta, old_delta in zip(new_delta_vals, old_delta_vals):
                    self.assertEqual(new_delta, expected_new_delta)
                    self.assertEqual(old_delta, expected_old_delta)

            if i in [75, 76]:
                expected_new_delta = INITIAL_DELTA * (DECAY_RATE**24)
                expected_old_delta = INITIAL_DELTA * (DECAY_RATE**24)
                for new_delta, old_delta in zip(new_delta_vals, old_delta_vals):
                    self.assertEqual(new_delta, expected_new_delta)
                    self.assertEqual(old_delta, expected_old_delta)

        # INITIAL_DELTA * (DECAY_RATE ** 25) should be the greater than 1.0, but
        # the delta value should be capped at 1.0
        for i in range(77, 100):
            ret_dict = ded_scheduler.step(model)
            new_delta_vals = ret_dict["new_delta_vals"]
            old_delta_vals = ret_dict["old_delta_vals"]
            self.assertEqual(len(new_delta_vals), 2)
            self.assertEqual(len(old_delta_vals), 2)

            expected_new_delta = 1.0
            expected_old_delta = (
                INITIAL_DELTA * (DECAY_RATE**24) if i == 77 else 1.0
            )
            for new_delta, old_delta in zip(new_delta_vals, old_delta_vals):
                self.assertEqual(new_delta, expected_new_delta)
                self.assertEqual(old_delta, expected_old_delta)

    def test_step_calculates_the_correct_new_delta_for_layers(self):
        CONJ_INITIAL_DELTA = 0.1
        CONJ_DECAY_DELAY = 5
        CONJ_DECAY_STEPS = 3
        CONJ_DECAY_RATE = 1.1

        DISJ_INITIAL_DELTA = 0.2
        DISJ_DECAY_DELAY = 2
        DISJ_DECAY_STEPS = 4
        DISJ_DECAY_RATE = 1.2

        # Create a neural DNF module
        model = NeuralDNF(3, 5, 4, 1.0)
        model.set_delta_val([CONJ_INITIAL_DELTA, DISJ_INITIAL_DELTA])

        # Create a scheduler that decays delta by a factor of 0.1 every 3 steps,
        # delayed by 5 steps
        conj_scheduler = DeltaDelayedExponentialDecayScheduler(
            initial_delta=CONJ_INITIAL_DELTA,
            delta_decay_delay=CONJ_DECAY_DELAY,
            delta_decay_steps=CONJ_DECAY_STEPS,
            delta_decay_rate=CONJ_DECAY_RATE,
            target_module_type=model.conjunctions.__class__.__name__,
        )
        disj_scheduler = DeltaDelayedExponentialDecayScheduler(
            initial_delta=DISJ_INITIAL_DELTA,
            delta_decay_delay=DISJ_DECAY_DELAY,
            delta_decay_steps=DISJ_DECAY_STEPS,
            delta_decay_rate=DISJ_DECAY_RATE,
            target_module_type=model.disjunctions.__class__.__name__,
        )

        # Expectations for the first 20 steps
        conj_expectations = [
            CONJ_INITIAL_DELTA,
            CONJ_INITIAL_DELTA,
            CONJ_INITIAL_DELTA,
            CONJ_INITIAL_DELTA,
            CONJ_INITIAL_DELTA,
            CONJ_INITIAL_DELTA * CONJ_DECAY_RATE,
            CONJ_INITIAL_DELTA * CONJ_DECAY_RATE,
            CONJ_INITIAL_DELTA * CONJ_DECAY_RATE,
            CONJ_INITIAL_DELTA * (CONJ_DECAY_RATE**2),
            CONJ_INITIAL_DELTA * (CONJ_DECAY_RATE**2),
            CONJ_INITIAL_DELTA * (CONJ_DECAY_RATE**2),
            CONJ_INITIAL_DELTA * (CONJ_DECAY_RATE**3),
            CONJ_INITIAL_DELTA * (CONJ_DECAY_RATE**3),
            CONJ_INITIAL_DELTA * (CONJ_DECAY_RATE**3),
            CONJ_INITIAL_DELTA * (CONJ_DECAY_RATE**4),
            CONJ_INITIAL_DELTA * (CONJ_DECAY_RATE**4),
            CONJ_INITIAL_DELTA * (CONJ_DECAY_RATE**4),
            CONJ_INITIAL_DELTA * (CONJ_DECAY_RATE**5),
            CONJ_INITIAL_DELTA * (CONJ_DECAY_RATE**5),
            CONJ_INITIAL_DELTA * (CONJ_DECAY_RATE**5),
        ]

        disj_expectations = [
            DISJ_INITIAL_DELTA,
            DISJ_INITIAL_DELTA,
            DISJ_INITIAL_DELTA * DISJ_DECAY_RATE,
            DISJ_INITIAL_DELTA * DISJ_DECAY_RATE,
            DISJ_INITIAL_DELTA * DISJ_DECAY_RATE,
            DISJ_INITIAL_DELTA * DISJ_DECAY_RATE,
            DISJ_INITIAL_DELTA * (DISJ_DECAY_RATE**2),
            DISJ_INITIAL_DELTA * (DISJ_DECAY_RATE**2),
            DISJ_INITIAL_DELTA * (DISJ_DECAY_RATE**2),
            DISJ_INITIAL_DELTA * (DISJ_DECAY_RATE**2),
            DISJ_INITIAL_DELTA * (DISJ_DECAY_RATE**3),
            DISJ_INITIAL_DELTA * (DISJ_DECAY_RATE**3),
            DISJ_INITIAL_DELTA * (DISJ_DECAY_RATE**3),
            DISJ_INITIAL_DELTA * (DISJ_DECAY_RATE**3),
            DISJ_INITIAL_DELTA * (DISJ_DECAY_RATE**4),
            DISJ_INITIAL_DELTA * (DISJ_DECAY_RATE**4),
            DISJ_INITIAL_DELTA * (DISJ_DECAY_RATE**4),
            DISJ_INITIAL_DELTA * (DISJ_DECAY_RATE**4),
            DISJ_INITIAL_DELTA * (DISJ_DECAY_RATE**5),
            DISJ_INITIAL_DELTA * (DISJ_DECAY_RATE**5),
        ]

        for i in range(20):
            conj_ret_dict = conj_scheduler.step(model.conjunctions)
            disj_ret_dict = disj_scheduler.step(model.disjunctions)

            conj_new_delta_vals = conj_ret_dict["new_delta_vals"]
            conj_old_delta_vals = conj_ret_dict["old_delta_vals"]
            disj_new_delta_vals = disj_ret_dict["new_delta_vals"]
            disj_old_delta_vals = disj_ret_dict["old_delta_vals"]

            self.assertEqual(len(conj_new_delta_vals), 1)
            self.assertEqual(len(conj_old_delta_vals), 1)
            self.assertEqual(len(disj_new_delta_vals), 1)
            self.assertEqual(len(disj_old_delta_vals), 1)

            for new_delta, old_delta in zip(
                conj_new_delta_vals, conj_old_delta_vals
            ):
                self.assertEqual(new_delta, conj_expectations[i])
                expected_old_delta = (
                    conj_expectations[i - 1] if i > 0 else CONJ_INITIAL_DELTA
                )
                self.assertEqual(old_delta, expected_old_delta)

            for new_delta, old_delta in zip(
                disj_new_delta_vals, disj_old_delta_vals
            ):
                self.assertEqual(new_delta, disj_expectations[i])
                expected_old_delta = (
                    disj_expectations[i - 1] if i > 0 else DISJ_INITIAL_DELTA
                )
                self.assertEqual(old_delta, expected_old_delta)

        # Conjunction:
        # CONJ_INITIAL_DELTA * (DECAY_RATE ** 10) should be achieved after 32
        # steps during i = 32 to 34
        # Disjunction:
        # DISJ_INITIAL_DELTA * (DECAY_RATE ** 8) should be the last delta value
        # below 1.0, and should be achieved after 30 steps during i = 30 to 33
        # DISJ_INITIAL_DELTA * (DECAY_RATE ** 9) should be the greater than 1.0,
        # but the delta value should be capped at 1.0. DECAY_RATE ** 9 should be
        # achieved after 34 steps
        for i in range(20, 38):
            conj_ret_dict = conj_scheduler.step(model.conjunctions)
            disj_ret_dict = disj_scheduler.step(model.disjunctions)

            conj_new_delta_vals = conj_ret_dict["new_delta_vals"]
            conj_old_delta_vals = conj_ret_dict["old_delta_vals"]
            disj_new_delta_vals = disj_ret_dict["new_delta_vals"]
            disj_old_delta_vals = disj_ret_dict["old_delta_vals"]

            self.assertEqual(len(conj_new_delta_vals), 1)
            self.assertEqual(len(conj_old_delta_vals), 1)
            self.assertEqual(len(disj_new_delta_vals), 1)
            self.assertEqual(len(disj_old_delta_vals), 1)

            if i == 30:
                expected_new_delta = DISJ_INITIAL_DELTA * (DISJ_DECAY_RATE**8)
                expected_old_delta = DISJ_INITIAL_DELTA * (DISJ_DECAY_RATE**7)
                for new_delta, old_delta in zip(
                    disj_new_delta_vals, disj_old_delta_vals
                ):
                    self.assertEqual(new_delta, expected_new_delta)
                    self.assertEqual(old_delta, expected_old_delta)

            if i in [31, 32, 33]:
                expected_new_delta = DISJ_INITIAL_DELTA * (DISJ_DECAY_RATE**8)
                expected_old_delta = DISJ_INITIAL_DELTA * (DISJ_DECAY_RATE**8)
                for new_delta, old_delta in zip(
                    disj_new_delta_vals, disj_old_delta_vals
                ):
                    self.assertEqual(new_delta, expected_new_delta)
                    self.assertEqual(old_delta, expected_old_delta)

            if i == 32:
                expected_new_delta = CONJ_INITIAL_DELTA * (CONJ_DECAY_RATE**10)
                expected_old_delta = CONJ_INITIAL_DELTA * (CONJ_DECAY_RATE**9)
                for new_delta, old_delta in zip(
                    conj_new_delta_vals, conj_old_delta_vals
                ):
                    self.assertEqual(new_delta, expected_new_delta)
                    self.assertEqual(old_delta, expected_old_delta)

            if i in [33, 34]:
                expected_new_delta = CONJ_INITIAL_DELTA * (CONJ_DECAY_RATE**10)
                expected_old_delta = CONJ_INITIAL_DELTA * (CONJ_DECAY_RATE**10)
                for new_delta, old_delta in zip(
                    conj_new_delta_vals, conj_old_delta_vals
                ):
                    self.assertEqual(new_delta, expected_new_delta)
                    self.assertEqual(old_delta, expected_old_delta)

            if i >= 34:
                expected_new_delta = 1.0
                expected_old_delta = (
                    DISJ_INITIAL_DELTA * (DISJ_DECAY_RATE**8)
                    if i == 34
                    else 1.0
                )
                for new_delta, old_delta in zip(
                    disj_new_delta_vals, disj_old_delta_vals
                ):
                    self.assertEqual(new_delta, expected_new_delta)
                    self.assertEqual(old_delta, expected_old_delta)

        # Conjunction:
        # CONJ_INITIAL_DELTA * (DECAY_RATE ** 24) should be the last delta value
        # below 1.0, and should be achieved after 74 steps during i = 74 to 76
        for i in range(38, 77):
            conj_ret_dict = conj_scheduler.step(model.conjunctions)
            disj_ret_dict = disj_scheduler.step(model.disjunctions)

            conj_new_delta_vals = conj_ret_dict["new_delta_vals"]
            conj_old_delta_vals = conj_ret_dict["old_delta_vals"]
            disj_new_delta_vals = disj_ret_dict["new_delta_vals"]
            disj_old_delta_vals = disj_ret_dict["old_delta_vals"]

            self.assertEqual(len(conj_new_delta_vals), 1)
            self.assertEqual(len(conj_old_delta_vals), 1)
            self.assertEqual(len(disj_new_delta_vals), 1)
            self.assertEqual(len(disj_old_delta_vals), 1)

            for new_delta, old_delta in zip(
                disj_new_delta_vals, disj_old_delta_vals
            ):
                self.assertEqual(new_delta, 1.0)
                self.assertEqual(old_delta, 1.0)

            if i == 74:
                expected_new_delta = CONJ_INITIAL_DELTA * (CONJ_DECAY_RATE**24)
                expected_old_delta = CONJ_INITIAL_DELTA * (CONJ_DECAY_RATE**23)
                for new_delta, old_delta in zip(
                    conj_new_delta_vals, conj_old_delta_vals
                ):
                    self.assertEqual(new_delta, expected_new_delta)
                    self.assertEqual(old_delta, expected_old_delta)

            if i in [75, 76]:
                expected_new_delta = CONJ_INITIAL_DELTA * (CONJ_DECAY_RATE**24)
                expected_old_delta = CONJ_INITIAL_DELTA * (CONJ_DECAY_RATE**24)
                for new_delta, old_delta in zip(
                    conj_new_delta_vals, conj_old_delta_vals
                ):
                    self.assertEqual(new_delta, expected_new_delta)
                    self.assertEqual(old_delta, expected_old_delta)

        # INITIAL_DELTA * (DECAY_RATE ** 25) should be the greater than 1.0, but
        # the delta value should be capped at 1.0
        for i in range(77, 100):
            conj_ret_dict = conj_scheduler.step(model.conjunctions)
            disj_ret_dict = disj_scheduler.step(model.disjunctions)

            conj_new_delta_vals = conj_ret_dict["new_delta_vals"]
            conj_old_delta_vals = conj_ret_dict["old_delta_vals"]
            disj_new_delta_vals = disj_ret_dict["new_delta_vals"]
            disj_old_delta_vals = disj_ret_dict["old_delta_vals"]

            self.assertEqual(len(conj_new_delta_vals), 1)
            self.assertEqual(len(conj_old_delta_vals), 1)
            self.assertEqual(len(disj_new_delta_vals), 1)
            self.assertEqual(len(disj_old_delta_vals), 1)

            for new_delta, old_delta in zip(
                disj_new_delta_vals, disj_old_delta_vals
            ):
                self.assertEqual(new_delta, 1.0)
                self.assertEqual(old_delta, 1.0)

            expected_old_delta = (
                CONJ_INITIAL_DELTA * (CONJ_DECAY_RATE**24) if i == 77 else 1.0
            )
            for new_delta, old_delta in zip(
                conj_new_delta_vals, conj_old_delta_vals
            ):
                self.assertEqual(new_delta, 1.0)
                self.assertEqual(old_delta, expected_old_delta)
