class Activator:
    def __init__(self):
        self.is_active = True

    def activate(self) -> None:
        self.is_active = True

    def deactivate(self) -> None:
        self.is_active = False
