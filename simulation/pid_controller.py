class PIDController:
    def __init__(self, P,I,D):
        self.P = P
        self.I=I
        self.D=D
        self.total_err = 0
        self.prev_err = 0
    
    def step(self, err):
        '''
        Returns control input
        '''
        self.total_err+=err
        err_diff = err-self.prev_err
        self.prev_err = err
        return self.P*err + self.I*self.total_err  - self.D*err_diff