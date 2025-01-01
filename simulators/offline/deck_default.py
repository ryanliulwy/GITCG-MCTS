# default deck from here:
# https://dottore-genius-invokation-tcg-simulator.readthedocs.io/en/stable/get-started.html

import dgisim as dg
from dgisim import FrozenDeck, MutableDeck, HashableDict
from dgisim import char
from dgisim import card

deck1 = dg.MutableDeck(
    chars=[char.Bennett, char.Klee, char.Keqing],
    cards={
        card.GrandExpectation: 2,
        card.PoundingSurprise: 2,
        card.ThunderingPenance: 2,
        card.Vanarana: 2,
        card.ChangingShifts: 2,
        card.LeaveItToMe: 2,
        card.SacrificialSword: 2,
        card.GamblersEarrings: 2,
        card.IHaventLostYet: 2,
        card.LotusFlowerCrisp: 2,
        card.NorthernSmokedChicken: 2,
        card.ElementalResonanceFerventFlames: 2,
        card.ElementalResonanceWovenFlames: 2,
        card.WindAndFreedom: 2,
        card.TeyvatFriedEgg: 2,
    }
)