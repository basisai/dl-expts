def levenshtein(u, v):
    prev = None
    curr = [0] + list(range(1, len(v) + 1))
    # Operations: (SUB, DEL, INS)
    prev_ops = None
    curr_ops = [(0, 0, i) for i in range(len(v) + 1)]
    for x in range(1, len(u) + 1):
        prev, curr = curr, [x] + ([None] * len(v))
        prev_ops, curr_ops = curr_ops, [(0, x, 0)] + ([None] * len(v))
        for y in range(1, len(v) + 1):
            delcost = prev[y] + 1
            addcost = curr[y - 1] + 1
            subcost = prev[y - 1] + int(u[x - 1] != v[y - 1])
            curr[y] = min(subcost, delcost, addcost)
            if curr[y] == subcost:
                (n_s, n_d, n_i) = prev_ops[y - 1]
                curr_ops[y] = (n_s + int(u[x - 1] != v[y - 1]), n_d, n_i)
            elif curr[y] == delcost:
                (n_s, n_d, n_i) = prev_ops[y]
                curr_ops[y] = (n_s, n_d + 1, n_i)
            else:
                (n_s, n_d, n_i) = curr_ops[y - 1]
                curr_ops[y] = (n_s, n_d, n_i + 1)
    return curr[len(v)], curr_ops[len(v)]


def scores(ref, hyp):
    cer_s, cer_i, cer_d, cer_n = 0, 0, 0, 0
    wer_s, wer_i, wer_d, wer_n = 0, 0, 0, 0
    for n in range(len(ref)):
        # update CER statistics
        _, (s, d, i) = levenshtein(ref[n], hyp[n])
        cer_s += s
        cer_d += d
        cer_i += i
        cer_n += len(ref[n])
        # update WER statistics
        _, (s, d, i) = levenshtein(ref[n].split(), hyp[n].split())
        wer_s += s
        wer_d += d
        wer_i += i
        wer_n += len(ref[n].split())

    if cer_n > 0:
        return (cer_s + cer_d + cer_i) / cer_n, (wer_s + wer_d + wer_i) / wer_n


def cer_scores(ref, hyp):
    # CER statistics
    _, (cer_s, cer_d, cer_i) = levenshtein(ref, hyp)
    cer_n = len(ref)
    if cer_n > 0:
        return cer_s / cer_n, cer_d / cer_n, cer_i / cer_n


def wer_scores(ref, hyp):
    # WER statistics
    _, (wer_s, wer_d, wer_i) = levenshtein(ref.split(), hyp.split())
    wer_n = len(ref.split())

    if wer_n > 0:
        return wer_s / wer_n, wer_d / wer_n, wer_i / wer_n
