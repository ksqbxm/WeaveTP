"""Decode policies: greedy (fail-stop) and teacher forcing (scale-up/down). The driver only calls the protocol."""


class Greedy:
    """argmax of the row's logits; a request ends after exactly output_len tokens, EOS ignored."""

    name = 'greedy'

    def next_tokens(self, logits, rows):
        return [int(logits[i].argmax()) for i in range(len(rows))]

    @staticmethod
    def is_done(row, generated):
        return generated >= row.output_len


class TeacherForcing:
    """The next input token is the reference output token; lengths are fixed the same way."""

    name = 'teacher_forcing'

    def __init__(self, reference_tokens):
        self.reference = reference_tokens  # seq -> output_token_ids

    def next_tokens(self, logits, rows):
        return [self.reference[row.seq][row.generated_so_far] for row in rows]

    @staticmethod
    def is_done(row, generated):
        return generated >= row.output_len


POLICIES = {'greedy': Greedy, 'teacher_forcing': TeacherForcing}
