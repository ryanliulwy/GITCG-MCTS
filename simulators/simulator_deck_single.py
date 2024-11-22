# strong single character deck
 
from dgisim import FrozenDeck, MutableDeck, HashableDict
from dgisim import char
from dgisim import card

deck = FrozenDeck(
    chars=(char.Mona, char.Klee, char.FatuiPyroAgent),
    cards=HashableDict({
        # Talent Cards
        card.PoundingSurprise: 2, # resource

        # Support Cards
        card.Katheryne: 2, # utility
        card.Vanarana: 2, # resource
        card.ParametricTransformer: 2, # resource

        # Arcane Legend
        card.CovenantOfRock: 1, # resource
        
        # Resonance Cards
        card.ElementalResonanceFerventFlames: 2, # damage
        card.ElementalResonanceWovenFlames: 2, # resource

        # Event Cards
        card.TheBestestTravelCompanion: 2, # resource
        card.ChangingShifts: 2, # resource
        card.TossUp: 2, # resource
        card.Strategize: 2, # resource
        card.IHaventLostYet: 1, # resource
        card.LeaveItToMe: 2, # utility
        card.HeavyStrike: 2, # damage

        # Food Cards
        card.NorthernSmokedChicken: 2, # resource
        card.TandooriRoastChicken: 2, # damage
    })
)