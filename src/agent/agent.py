class Agent:
    def __init__(self, state_size, action_size=3):
        self.state_size = state_size
        self.action_size = action_size
        self.inventory = []
        self.memory = []

    def act(self, state):
        raise NotImplementedError

    def remember(self, state, action, reward, next_state, done):
        # 기본적으로 메모리에 저장
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self, batch_size):
        # 기본적으로 아무것도 하지 않음
        pass

    def train(self):
        # 기본적으로 아무것도 하지 않음
        pass
