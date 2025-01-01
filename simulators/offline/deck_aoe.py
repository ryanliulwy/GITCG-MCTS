# strong aoe dmg deck

from dgisim import FrozenDeck, MutableDeck, HashableDict
from dgisim import char
from dgisim import card

deck1 = FrozenDeck(
    chars = (char.Fischl, char.Keqing, char.Ganyu),
    cards=HashableDict({ # chat gpt-ed
        # Core Cards
        card.ThunderSummonersCrown: 2,
        card.BlizzardStrayer: 1,
        card.ThunderingFury: 2,
        card.ElementalResonanceHighVoltage: 2,
        card.ElementalResonanceShatteringIce: 1,

        # Weapons
        card.SacrificialBow: 1,
        card.AmosBow: 1,
        card.SacrificialSword: 1,

        # Utility Cards
        card.Strategize: 2,
        card.QuickKnit: 2,
        card.Starsigns: 2,
        card.TossUp: 2,

        # Reaction Amplifiers
        card.StellarPredator: 1,
        card.ProliferatingSpores: 2,

        # Support Cards
        card.Liben: 1,
        card.Paimon: 1,
        card.TreasureSeekingSeelie: 1,

        # Food Buffs
        card.SweetMadame: 1,
        card.MintyMeatRolls: 1,

        # Event Cards
        card.ThunderAndEternity: 2,
        card.JoyousCelebration: 1,
        card.LeaveItToMe: 1, 
    })
)