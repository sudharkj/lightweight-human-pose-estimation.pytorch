from modules.pose_state import PoseState


class ActivityFSM:
    def __init__(self, activity_json):
        self.states = []
        for pose_json in activity_json:
            self.states.append(PoseState(pose_json))
        if len(self.states) > 0:
            self.current = 0
        else:
            self.current = -1

    def update(self, current_pose):
        if len(self.states) < self.current < 0:
            return True, False, "Error occured!!!"

        success, info = self.states[self.current].is_reached(current_pose)
        if success:
            self.current += 1

        return self.current == len(self.states), success, info
