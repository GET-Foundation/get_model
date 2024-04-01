
@dataclass
class BaseTaskConfig:
    model: str = MISSING
    metadata: str = MISSING


@dataclass
class MutationTaskConfig(BaseTaskConfig):
    mutation_file: str = MISSING


@dataclass
class PeakInactivationTaskConfig(BaseTaskConfig):
    peak_file: str = MISSING


class BaseTask:
    def __init__(self, cfg: BaseTaskConfig):
        self.cfg = cfg
        self.load_metadata()

    def load_metadata(self):
        metadata = pd.read_csv(self.cfg.metadata)
        self.metadata = metadata

    def predict(self):
        pass

    def plot(self):
        pass


class MutationTask(BaseTask):
    def __init__(self, cfg: MutationTaskConfig):
        super().__init__(cfg)
        self.load_mutation_file()

    def load_mutation_file(self):
        mutations = pd.read_csv(self.mutation_file)
        self.mutations = mutations

    def wt_dataloader(self):
        pass

    def mut_dataloader(self):
        pass

    def predict(self, lm: LitModel):
        wt_predictions = []
        mut_predictions = []
        for batch in self.wt_dataloader:
            wt_predictions.append(lm.model(batch))

        for batch in self.mut_dataloader:
            mut_predictions.append(lm.model(batch))

    def plot(self):
        pass
