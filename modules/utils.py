ARCHITECTURES = ['SQU', 'SYM', 'ENC']


def get_units(idx, neurons, architecture, layers=None):
    assert architecture in ARCHITECTURES
    if architecture == 'SQU':
        return neurons
    elif architecture == 'SYM':
        assert (layers is not None and layers > 2)
        if layers % 2 == 1:
            return neurons * 2 ** (int(layers / 2) - abs(int(layers / 2) - idx))
        else:
            x = int(layers / 2)
            idx = idx if idx < x else 2 * x - idx - 1
            return neurons * 2 ** (int(layers / 2) - abs(int(layers / 2) - idx))

    elif architecture == 'ENC':
        assert (layers is not None and layers > 2)
        if layers % 2 == 0:
            x = int(layers / 2)
            idx = idx if idx < x else 2 * x - idx - 1
            return neurons * 2 ** (int(layers / 2) - 1 - idx)
        else:
            x = int(layers / 2)
            idx = idx if idx < x else 2 * x - idx
            return neurons * 2 ** (int(layers / 2) - idx)