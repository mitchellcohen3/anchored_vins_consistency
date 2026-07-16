import typing
from navlie.lib.states import State

def shift_stamps(state_list: typing.List[State]):
    init_stamp = state_list[0].stamp
    for state in state_list:
        state.stamp -= init_stamp