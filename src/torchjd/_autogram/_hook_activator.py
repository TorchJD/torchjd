class HookActivator:
    def __init__(self):
        self.state = True

    def activate(self) -> None:
        self.state = True

    def deactivate(self) -> None:
        self.state = False
