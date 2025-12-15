#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# DISTRIBUTION STATEMENT A: Approved for public release. Distribution is
# unlimited.
# ----------------------------------------------------------------------------
"""Functions to calculate skill score statistics."""

import numpy as np


def state_check(truth_vals, test_vals, event_val='eia'):
    """Calculate the skill state of an observation against truth.

    Parameters
    ----------
    truth_vals : array-like
        Array of values to compare to `event_val` specifying the truth status
        to test against
    test_vals : array-like
        Array of values to compare to `event_val` specifying the data status to
        test against the truth
    event_val : str, int, or float
        Value specifying the state indicating a positive event (default='eia')

    Returns
    -------
    event_states : array-like
        Categories of skill score states corresponding to the combination of
        test and truth values for the desired event value

    Raises
    ------
    ValueError
        If `test_vals` and `truth_vals` are different lengths

    Notes
    -----
    The skill score states are:
    - H: hit
    - M: miss
    - C: correct negative
    - F: false alarm

    """
    # Test the inputs
    if len(test_vals) != len(truth_vals):
        raise ValueError('Number of test values are not equal to truth values')

    # Initialize the output
    event_states = []

    # Cycle through each test and truth value
    for i, test in enumerate(test_vals):
        truth = truth_vals[i]

        # Determine the possible outcomes based on the truth observation
        # of the desired event
        if truth == event_val:
            # This is either a hit or a miss, depending on whether or not the
            # test value agrees with the truth value
            if test == event_val:
                event_states.append('H')
            else:
                event_states.append('M')
        else:
            # This is either a correct negative or false alarm, depending on
            # whether or not the test value agrees with the truth value
            if test != event_val:
                event_states.append('C')
            else:
                event_states.append('F')

    event_states = np.array(event_states)

    return event_states


def coin_toss_state(event_states):
    """Calcualte the event states needed for a coin toss outcome.

    Parameters
    ----------
    event_states : array-like
        Array of event states 'H', 'M', 'F', and 'C' that will be used to
        create a coin-toss model of the same size and proportion of truth
        states

    Returns
    -------
    hit : int
        Number of hits for the coin toss model
    miss : int
        Number of misses for the coin toss model
    falarm : int
        Number of false alarms for the coin toss model
    corneg : int
        Number of correct negatives for the coin toss model

    """
    # Determine the number of times the truth model was in the desired state
    coin_hm = sum((event_states == 'H') | (event_states == 'M'))

    # Determine the number of times the truth model was not in the desired state
    coin_fc = sum((event_states == 'M') | (event_states == 'C'))

    # The hit/miss/false-alarm/correct-negative of a coin is half of hit + miss
    # for hits and misses and half of false-alarm + correct-negative for
    # correct negatives and false alarms
    hit = int(np.floor(coin_hm / 2))
    miss = int(coin_hm) - hit
    corneg = int(np.floor(coin_fc / 2))
    falarm = coin_fc - corneg

    return hit, miss, corneg, falarm


def liemohn_skill_score(event_states, coin=False):
    """Calcualte the Liemohn skill scores using the skill event states.

    Parameters
    ----------
    event_states : array-like
        array of event states 'H', 'M', 'F', and 'C'
    coin : bool
        if True, returns will be LSS for a coin
        if False, returns will be LSS of event_states (default)
    Returns
    -------
    LSS1 : double
        Liemohn Skill Score 1
    LSS2 : double
        Liemohn Skill Score 2
    LSS3 : double
        Liemohn Skill Score 3
    LSS4 : double
        Liemohn Skill Score 4

    References
    ----------
    Liemohn 2025 under review

    See Also
    --------
    state_check

    """
    if coin:
        # Determine the state sums for a coin toss model
        hit, miss, falarm, corneg = coin_toss_state(event_states)
    else:
        # Determine the state sums for each skill quadrant
        hit = sum(event_states == 'H')
        falarm = sum(event_states == 'F')
        miss = sum(event_states == 'M')
        corneg = sum(event_states == 'C')

    # Liemohn Skill Score 1
    lss1 = ((2 * hit * corneg + miss * corneg + hit * falarm - hit * miss
             - miss ** 2 - falarm ** 2 - falarm * corneg)
            / (2 * (hit + miss) * (falarm + corneg)))

    # Liemohn Skill Score 2 (LSS2t/LSS2b)
    lss2t = (hit * ((hit + miss) ** 2 + 2 * (hit + miss) * (falarm + corneg))
             - (hit + miss) ** 2 * (hit + miss + falarm))
    lss2b = ((hit + miss + falarm) * ((hit + miss) ** 2 + 2 * (hit + miss)
                                      * (falarm + corneg))
             - (hit + miss) ** 2 * (hit + miss + falarm))
    lss2 = lss2t / lss2b

    # Liemohn Skill Score 3
    lss3 = ((hit + corneg) - (miss + falarm)) / (hit + miss + falarm + corneg)

    # Liemohn Skill Score 4
    lss4 = ((hit * (2 * (hit + miss) + falarm + corneg) - (hit + miss)
             * (hit + miss + falarm))
            / ((hit + miss + falarm) * (2 * (hit + miss) + falarm + corneg)
               - (hit + miss) * (hit + miss + falarm)))

    return lss1, lss2, lss3, lss4


def calc_pc_and_csi(event_states, coin=False):
    """Calculate the percent correct and critical success index.

    Parameters
    ----------
    event_states : array-like
        array of event states 'H', 'M', 'F', and 'C'
    coin : bool
        if True, returns will be LSS for a coin
        if False, returns will be LSS of event_states

    Returns
    -------
    pcor : double
        percent correct as a decimal between 0 and 1
    cs_ind : double
        critical success index as a decimal between 0 and 1

    """
    if coin:
        # Determine the state sums for a coin toss model
        hit, miss, falarm, corneg = coin_toss_state(event_states)
    else:
        # Determine the state sums for each skill quadrant
        hit = sum(event_states == 'H')
        falarm = sum(event_states == 'F')
        miss = sum(event_states == 'M')
        corneg = sum(event_states == 'C')

    pcor = (hit + corneg) / (hit + miss + falarm + corneg)
    cs_ind = hit / (hit + miss + falarm)

    return pcor, cs_ind
