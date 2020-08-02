import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from os import listdir
import mido
from mido import MidiFile
import numpy as np
import copy
import math
import matplotlib.pyplot as plt

# для возможности прослушивания midi
try:
    HEARING_PORT = mido.open_output()
except:
    pass

np.set_printoptions(threshold=np.nan)

BASE_NOTE = 21  # номер нижней клавиши на реальной фортепианной клавиатуре
KEYS = 88  # клавиш на клавиатуре
FREQ = 4  # ticks for beat

# класс мидифайла песни
class Song:
    # Если в качестве name подана строка, то будет загружен миди файл с таким названием
    # Помимо строки может принимать массив, где числа от 0 до 88 соотв. нотам, иначе пауза

    def __init__(self, data, finished=True, hold_mode=True):
        self.correct = True  # пометка о том, что файл загрузился без ошибок.
        self.hold_mode = hold_mode

        if isinstance(data, str):
            self.name = data
            self.notes = np.zeros((32, KEYS), dtype=int)
            self.tempo_changes = 0

            # переводим в ноты
            try:
                self.mid = MidiFile(data)
            except:
                print(self.name, ": Error opening file with mido")
                self.correct = False
                return

            for msg in self.mid:
                if msg.type == 'time_signature':
                    if msg.denominator not in {2, 4, 8, 16} or msg.numerator not in {2, 4, 8, 16}:
                        print(self.name, ": Error: bad signature ", msg.numerator, '/', msg.denominator)
                        self.correct = False
                        return
                if msg.type == 'set_tempo':
                    self.tempo_changes += 1

            data_tracks = set()
            for t_id, track in enumerate(self.mid.tracks):
                absolute_time_passed = 0
                pressed_keys = np.zeros((88))

                for msg in track:
                    absolute_time_passed += msg.time

                    key_on = (msg.type == "note_on" and msg.velocity > 0)
                    key_off = (msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0))
                    if key_on or key_off:
                        data_tracks.add(t_id)
                        t = round(absolute_time_passed * FREQ / self.mid.ticks_per_beat)

                        while t >= len(self.notes):
                            if self.hold_mode:
                                self.notes = np.vstack([self.notes, np.zeros((32, 88), dtype=int)])
                            else:
                                self.notes = np.vstack([self.notes, np.tile(self.notes[-1][None], (32, 1))])

                        if (msg.note - BASE_NOTE < 0 or msg.note - BASE_NOTE >= 88):
                            print(self.name, ": ERROR: note out of range, ", msg.note, msg.note - BASE_NOTE)
                            self.correct = False
                            return

                        if self.hold_mode:
                            self.notes[t, msg.note - BASE_NOTE] = msg.velocity * key_on
                        else:
                            self.notes[t:, msg.note - BASE_NOTE] = msg.velocity * key_on

            if len(data_tracks) > 2:
                print(self.name, ": Error: must be one (or two for two hands) track only")
                self.correct = False
                return

            self.notes = self.notes[self.notes.sum(axis=1).nonzero()[0][0]:]
        else:
            self.name = "Generated song!"
            self.notes = np.zeros((0, KEYS))

            self.mid = MidiFile(type=0)
            self.track = mido.MidiTrack()
            self.mid.tracks.append(self.track)

            self.time_passed = 0
            self.release = []

            for line in data:
                self.add(line)

            if finished:
                self.finish()

    # проиграть песню
    def play(self):
        for msg in self.mid.play():
            HEARING_PORT.send(msg)

    # транспонирование мелодии
    def transpose(self, shift):
        if (not self.correct or
                (shift > 0 and self.notes[:, -shift:].sum() != 0) or
                (shift < 0 and self.notes[:, :-shift].sum() != 0)):
            return False

        self.notes = np.hstack([self.notes[:, -shift:], self.notes[:, :-shift]])
        return True

    def drawSong(song, scale=(None, None)):
        if scale[0] is None:
            scale = (song.notes.shape[0] / 10, song.notes.shape[1] / 10)

        plt.figure(figsize=scale)
        plt.title(song.name)
        plt.imshow(song.notes.T, aspect='auto', origin='lower')
        plt.xlabel("time")
        plt.xticks(np.arange(0, song.notes.shape[0], 4))
        plt.show()

    # добавление новых строк в ноты
    def add(self, played):
        self.notes = np.vstack([self.notes, played])

        # "отпуск" нот, сыгранных долю назад
        if self.hold_mode and played.sum() > 0:
            for i in self.release:
                self.track.append(mido.Message('note_on', note=BASE_NOTE + i, velocity=0, time=self.time_passed))
                self.time_passed = 0
            self.release = []

        # добавление новых нот
        for i, key in enumerate(played):
            if key > 0:
                if i not in self.release:
                    self.track.append(mido.Message('note_on', note=BASE_NOTE + i, velocity=key, time=self.time_passed))
                    self.time_passed = 0
                    self.release.append(i)
            elif not self.hold_mode and i in self.release:
                self.track.append(mido.Message('note_on', note=BASE_NOTE + i, velocity=0, time=self.time_passed))
                self.time_passed = 0
                self.release.remove(i)

        self.time_passed += self.mid.ticks_per_beat // FREQ

    # должна быть вызвана в конце создания файла (?) работает
    def finish(self):
        for i in self.release:
            self.track.append(mido.Message('note_off', note=BASE_NOTE + i, velocity=64, time=self.time_passed))
            self.time_passed = 0
        self.track.append(mido.Message('note_off', note=BASE_NOTE, velocity=0, time=self.time_passed))

    # сохранение
    def save_file(self, name):
        self.mid.save(name + '.mid')

"""МОДЕЛЬ"""
# периоды повторения, которые будут искаться
HISTORY_TIMES = np.array([4, 8, 16, 32, 48, 64, 96, 128])

# время (в двоичном коде) в качестве дополнительного ввода
TIME_AS_INPUT_SIZE = 6

# количество голосов
VOICES = 5

# Параметры сети
num_input = 12 + TIME_AS_INPUT_SIZE  # измерение одного входа в данный момент
num_hidden_local = 100  # измерение LSTM для локальных зависимостей
num_hidden_read = 100  # измерение LSTM для чтения из истории
num_hidden_aggreg = 130  # измерение уровня агрегации
num_hidden_voicegen = 100  # измерение уровня генерации голосов
num_output = 12  # выходное измерение для каждого голоса

num_hidden_decoder = 100  # размерность декодера LSTM
num_decoder_output = 88  # размер выходного сигнала декодера
num_decoder_rbm_hidden = 36  # размерность скрытого слоя декодера rbm
rbm_k = 1  # итерации выборки Гиббса в декодере

INDICES = Variable(torch.LongTensor(HISTORY_TIMES - 1))


class History(nn.Module):
    '''просстая модель внимания'''

    def __init__(self):
        super(History, self).__init__()

        self.read_lstm = nn.LSTM(num_input, num_hidden_read)
        self.read_linear = nn.Linear(num_hidden_read, len(HISTORY_TIMES))

    def init_hidden(self, batch_size, history_init=None):
        self.hidden = (Variable(torch.zeros(1, batch_size, num_hidden_read)),
                       Variable(torch.zeros(1, batch_size, num_hidden_read)))

        # хранение истории последовательности, требует инициализации
        self.history = history_init

    def forward(self, x):
        # получение коэффициента взвешенной суммы
        read_index, self.hidden = self.read_lstm(x, self.hidden)
        read_index = F.softmax(self.read_linear(read_index), dim=2)
        self.read_index = read_index.data[0].numpy()

        # обновление истории и выбор только тех, у которых есть потенциальные музыкальные периоды
        self.history = torch.cat([x[0][:, None, :num_output], self.history[:, :-1]], dim=1)
        variants = torch.index_select(self.history, 1, INDICES)

        # расчет взвешенной суммы
        read = torch.bmm(read_index[0, :, None], variants).squeeze(1)

        return read[None]


class PSampler(nn.Module):
    '''Сэмплер поголосовой полифонии'''

    def __init__(self):
        super(PSampler, self).__init__()

        # все голоса lstms, параметры общие
        self.lstm = nn.LSTM(num_hidden_aggreg + 2 * num_output, num_hidden_voicegen)
        self.linear = nn.Linear(num_hidden_voicegen, 2 * num_output)

    def init_hidden(self, batch_size):
        self.hidden = []
        for i in range(VOICES):
            self.hidden.append((Variable(torch.zeros(1, batch_size, num_hidden_voicegen)),
                                Variable(torch.zeros(1, batch_size, num_hidden_voicegen))))

    def forward(self, x, next_x=None):
        # для хранения решений, какие ноты будут играть, а какие нет, и все вероятности
        sampled_notes = torch.zeros(1, x.size()[1], num_output)
        sample_p = Variable(torch.zeros(1, x.size()[1], num_output))
        neg_sample_p = Variable(torch.ones(1, x.size()[1], num_output))

        banned_notes = torch.zeros(1, x.size()[1], num_output)
        ban_p = Variable(torch.zeros(1, x.size()[1], num_output))
        neg_ban_p = Variable(torch.ones(1, x.size()[1], num_output))

        # для визуализации
        self.voice_distributions = []
        self.voice_decisions = []

        for i in range(VOICES):
            # forward pass
            all_input = torch.cat([x, Variable(sampled_notes), Variable(banned_notes)], dim=2)
            out, self.hidden[i] = self.lstm(all_input, self.hidden[i])
            out = self.linear(out)

            # выходов для нот, решения по которым уже приняты, должен быть равен нулю
            coeff = (1 - torch.cat([sampled_notes, sampled_notes], dim=2)) * \
                    (1 - torch.cat([banned_notes, banned_notes], dim=2))
            out = Variable(coeff) * torch.exp(out)
            out = out / out.sum(2)[:, :, None]

            # сохранение результата
            self.voice_distributions.append(out.data[0].numpy())

            # вероятность сэмплирования - это вероятность сэмплирования в предыдущих тембрах плюс выборка в текущем тембре
            sample_p = sample_p + neg_sample_p * out[:, :, :num_output]
            neg_sample_p = neg_sample_p * (1 - out[:, :, :num_output])  # probability of NOT to sample note

            # то же самое для запрета ноты
            ban_p = ban_p + neg_ban_p * out[:, :, num_output:]
            neg_ban_p = neg_ban_p * (1 - out[:, :, num_output:])

            # принятие решения (отбор проб)
            sample = torch.LongTensor(x.size()[1], 2 * num_output).zero_()
            sample.scatter_(1, torch.multinomial(out.squeeze(0).data, 1), 1)

            self.voice_decisions.append(sample.numpy())

            # мы требуем  ground truth в качестве входных данных для голосов на этапе обучения
            if self.training:
                sample = torch.cat([(sample[:, :num_output] + sample[:, num_output:]) * next_x,
                                    (sample[:, :num_output] + sample[:, num_output:]) * (1 - next_x)], dim=1)

            # добавление решения к окончательному ответу
            sampled_notes[0] += sample[:, :num_output].float()
            banned_notes[0] += sample[:, num_output:].float()

        return sampled_notes, sample_p * (1 - ban_p)


import itertools


def time_generator():
    times = [np.array(t) for t in itertools.product([0, 1], repeat=TIME_AS_INPUT_SIZE)]
    t = 0

    while True:
        yield torch.FloatTensor(times[t])
        t = (t + 1) % len(times)


class Model(nn.Module):
    """Генератор состоит из параллельного LSTM
     и HistoryUser в качестве первого уровня и LSTM
     в качестве вывода второго уровня, отправляемого в полифонический сэмплер."""

    def __init__(self):
        super(Model, self).__init__()

        self.local_lstm = nn.LSTM(num_input, num_hidden_local)
        self.history_reader = History()
        self.aggregation_lstm = nn.LSTM(num_output + num_hidden_local, num_hidden_aggreg)
        self.sampler = PSampler()

        self.init_hidden(1, None)

    def init_hidden(self, batch_size, history_init):
        self.timer = time_generator()

        self.local_hidden = (Variable(torch.zeros(1, batch_size, num_hidden_local)),
                             Variable(torch.zeros(1, batch_size, num_hidden_local)))

        self.history_reader.init_hidden(batch_size, history_init)

        self.aggreg_hidden = (Variable(torch.zeros(1, batch_size, num_hidden_aggreg)),
                              Variable(torch.zeros(1, batch_size, num_hidden_aggreg)))

        self.sampler.init_hidden(batch_size)

    def forward(self, x, next_x=None):
        inp = torch.cat([x, Variable(self.timer.__next__())[None, None].repeat(1, x.size()[1], 1)], dim=2)

        local_out, self.local_hidden = self.local_lstm(inp, self.local_hidden)
        read = self.history_reader(inp)

        out, self.aggreg_hidden = self.aggregation_lstm(torch.cat([local_out, read], dim=2), self.aggreg_hidden)

        return self.sampler(out, next_x)


class Decoder(nn.Module):
    """декодер и генератор скорости"""

    def __init__(self):
        super(Decoder, self).__init__()

        # LSTM-RBM в качестве декодера
        self.decoder_lstm = nn.LSTM(num_output + num_decoder_output, num_hidden_decoder)
        self.v_bias_linear = nn.Linear(num_hidden_decoder, num_decoder_output)
        self.h_bias_linear = nn.Linear(num_hidden_decoder, num_decoder_rbm_hidden)

        # параметр RBM
        self.W = nn.Parameter(torch.randn(num_decoder_rbm_hidden, num_decoder_output) * 1e-2)

        # скорость, генерируемая LSTM
        self.velocity_lstm = nn.LSTM(num_output + num_decoder_output, num_hidden_decoder)
        self.velocity_linear = nn.Linear(num_hidden_decoder, num_decoder_output)
        self.init_hidden(1)

    def init_hidden(self, batch_size):
        self.decoder_hidden = (Variable(torch.zeros(1, batch_size, num_hidden_decoder)),
                               Variable(torch.zeros(1, batch_size, num_hidden_decoder)))
        self.velocity_hidden = (Variable(torch.zeros(1, batch_size, num_hidden_decoder)),
                                Variable(torch.zeros(1, batch_size, num_hidden_decoder)))

    # так было сделано в одной из реализаций rbm ...
    # кажется законным, хотя и нелепым
    def sample_from_p(self, p):
        return F.relu(torch.sign(p - Variable(torch.rand(p.size()))))

    # для оптимизации во время обучения
    def free_energy(self, v, v_bias, h_bias):
        vbias_term = (v * v_bias).sum(2)
        hidden_term = (v.bmm(self.W.t()[None]) + h_bias).exp().add(1).log().sum(2)
        return (-hidden_term - vbias_term).mean()

    def forward(self, x, prev_x_decoded):
        inp = torch.cat([x, prev_x_decoded], dim=2)

        out, self.decoder_hidden = self.decoder_lstm(inp, self.decoder_hidden)
        vel, self.velocity_hidden = self.velocity_lstm(inp, self.velocity_hidden)

        return F.sigmoid(self.velocity_linear(vel)), self.v_bias_linear(out), self.h_bias_linear(out)

    # Выборка Гиббса из RBM
    def sample(self, _v, v_bias, h_bias):
        v = _v

        for _ in range(rbm_k):
            p_h = F.sigmoid(v.bmm(self.W.t()[None]) + h_bias)
            h = self.sample_from_p(p_h)

            p_v = F.sigmoid(h.bmm(self.W[None]) + v_bias)
            v = self.sample_from_p(p_v)
        return v, p_v


def sample(model, decoder, start, decoder_start, time_limit):
    '''generation from model'''
    # initialization
    model.init_hidden(1, Variable(torch.zeros(1, int(HISTORY_TIMES.max()), num_output)))
    decoder.init_hidden(1)
    model.eval()
    decoder.eval()


    sample = [start[0]]
    decoded_sample = [decoder_start[0] / 128.]

    for t in range(len(start) - 1):
        inp = Variable(torch.FloatTensor(start[t])[None, None])
        next_notes, _ = model(inp)

        decoder_inp = Variable(torch.FloatTensor(start[t + 1])[None, None]), \
                      Variable(torch.FloatTensor(decoder_start[t + 1] / 128.)[None, None])
        decoder(*decoder_inp)

        sample.append(start[t + 1])
        decoded_sample.append(decoder_start[t + 1] / 128.)

    next_notes = torch.FloatTensor(start[-1][None, None])

    # generation cycle
    while len(sample) < time_limit:
        # 12-нотное представление
        next_notes, _ = model(Variable(next_notes))
        sample.append(next_notes[0][0].numpy())

        # декодирование
        prev_decoded = Variable(torch.FloatTensor((decoded_sample[-1] > 0).astype(np.float32))[None, None])
        velocity, v_bias, h_bias = decoder(Variable(next_notes), prev_decoded)
        next_decoded, _ = decoder.sample(prev_decoded, v_bias, h_bias)

        # генерация скорости и применение маски из 12 нот
        mask = np.tile(sample[-1], (1 + num_decoder_output // num_output))[:num_decoder_output]
        velocity = velocity[0][0].data.numpy()
        velocity[np.logical_not(mask * next_decoded[0][0].data.numpy())] = 0
        decoded_sample.append(velocity)

    return np.array(sample), (128 * np.array(decoded_sample)).astype(int)


def compress(song):
    '''сжатие в 12-нотном представлении'''
    compressed = song[:, :12] > 0
    for i in range(1, 7):
        compressed = np.logical_or(compressed, song[:, 12 * i:12 * i + 12] > 0)
    return compressed


# loading model
model = Model()
decoder = Decoder()

model.load_state_dict(torch.load("model.pt"))
decoder.load_state_dict(torch.load("decoder.pt"))

seeds = np.load("dataset.npy")

seed = seeds[np.random.randint(0, len(seeds))]
print(seed)
print(np.shape(seed))
print(compress(seed).astype(np.float32))
print(np.shape(compress(seed).astype(np.float32)))
cgen, gen = sample(model, decoder, compress(seed).astype(np.float32), seed.astype(np.float32), 256)

Song(gen.astype(int), finished=True).save_file("sample")
