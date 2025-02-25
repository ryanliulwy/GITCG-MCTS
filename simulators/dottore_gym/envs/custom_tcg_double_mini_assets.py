# implement kaeya
# implement sword
# implement food


class Kaeya(Character):
    # basic info
    _ELEMENT = Element.CRYO
    _WEAPON_TYPE = WeaponType.SWORD
    _TALENT_STATUS = stt.ColdBloodedStrikeStatus
    _FACTIONS = frozenset((Faction.MONDSTADT,))

    _SKILL1_COST = AbstractDice({
        Element.CRYO: 1,
        Element.ANY: 2,
    })
    _SKILL2_COST = AbstractDice({
        Element.CRYO: 3,
    })
    _ELEMENTAL_BURST_COST = AbstractDice({
        Element.CRYO: 4,
    })

    def _skill1(self, game_state: GameState, source: StaticTarget) -> tuple[eft.Effect, ...]:
        return normal_attack_template(
            game_state=game_state,
            source=source,
            element=Element.PHYSICAL,
            damage=2,
        )

    def _skill2(self, game_state: GameState, source: StaticTarget) -> tuple[eft.Effect, ...]:
        return (
            eft.ReferredDamageEffect(
                source=source,
                target=DynamicCharacterTarget.OPPO_ACTIVE,
                element=Element.CRYO,
                damage=3,
                damage_type=DamageType(elemental_skill=True),
            ),
        )

    def _elemental_burst(self, game_state: GameState, source: StaticTarget) -> tuple[eft.Effect, ...]:
        return (
            eft.EnergyDrainEffect(
                target=source,
                drain=self.max_energy,
            ),
            eft.ReferredDamageEffect(
                source=source,
                target=DynamicCharacterTarget.OPPO_ACTIVE,
                element=Element.CRYO,
                damage=1,
                damage_type=DamageType(elemental_burst=True),
            ),
            eft.OverrideCombatStatusEffect(
                target_pid=source.pid,
                status=stt.IcicleStatus(),
            )
        )

    @classmethod
    def from_default(cls, id: int = -1) -> Self:
        return cls(
            id=id,
            alive=True,
            hp=10,
            max_hp=10,
            energy=0,
            max_energy=2,
            hiddens=stts.Statuses(()),
            statuses=stts.Statuses(()),
            elemental_aura=ElementalAura.from_default(),
        )


# mondstadt hash brown (food || +2 hp, 1 cost)
class MondstadtHashBrown():
    @override
    @classmethod
    def heal_amount(cls) -> int:
        return 2
    
# broken rimes echo (cryo artifact)
class BrokenRimesEcho(ArtifactEquipmentCard):
    _DICE_COST = AbstractDice({Element.ANY: 2})
    ARTIFACT_STATUS = stt.BrokenRimesEchoStatus

@dataclass(frozen=True, kw_only=True) # stt.BrokenRimesEchoStatus
class BrokenRimesEchoStatus(_ElementalDiscountStatus):
    _ELEMENT: ClassVar[Element] = Element.CRYO

    @cached_classproperty
    def CARD(cls) -> type[crd.ArtifactEquipmentCard]:
        from ..card.card import BrokenRimesEcho
        return BrokenRimesEcho
    
# sweet madame (food || +1 hp, 0 cost)
class SweetMadame(_DirectHealCard, _CharTargetChoiceProvider):
    _DICE_COST = AbstractDice({})

    @override
    @classmethod
    def heal_amount(cls) -> int:
        return 1

# aquila favonia sword 
class AquilaFavonia(WeaponEquipmentCard):
    _DICE_COST = AbstractDice({Element.OMNI: 3})
    WEAPON_TYPE = WeaponType.SWORD
    WEAPON_STATUS = stt.AquilaFavoniaStatus

# aquila favonia sword status
@dataclass(frozen=True, kw_only=True)
class AquilaFavoniaStatus(WeaponEquipmentStatus, _UsageLivingStatus):
    WEAPON_TYPE: ClassVar[WeaponType] = WeaponType.SWORD
    usages: int = 2
    MAX_USAGES: ClassVar[int] = 2
    activated: bool = False
    HP_RECOVERY: ClassVar[int] = 1

    REACTABLE_SIGNALS: ClassVar[frozenset[TriggeringSignal]] = frozenset((
        TriggeringSignal.POST_SKILL,
        TriggeringSignal.ROUND_END,
    ))

    @cached_classproperty
    def CARD(cls) -> type[crd.WeaponEquipmentCard]:
        from ..card.card import AquilaFavonia
        return AquilaFavonia

    @override
    def _inform(
            self,
            game_state: GameState,
            status_source: StaticTarget,
            info_type: Informables,
            information: InformableEvent,
    ) -> Self:
        if info_type is Informables.POST_SKILL_USAGE:
            assert isinstance(information, SkillIEvent)
            if (
                    self.usages > 0
                    and not self.activated
                    and information.source.pid is status_source.pid.other
            ):
                return replace(self, activated=True)
        return self

    @override
    def _react_to_signal(
            self, game_state: GameState, source: StaticTarget, signal: TriggeringSignal,
            detail: None | InformableEvent
    ) -> tuple[list[eft.Effect], None | Self]:
        if signal is TriggeringSignal.POST_SKILL and self.activated:
            if self._target_is_self_active(game_state, source, source):
                return [
                    eft.RecoverHPEffect(
                        source=source,
                        target=source,
                        recovery=self.HP_RECOVERY,
                    ),
                ], replace(self, usages=-1, activated=False)
            else:
                return [], replace(self, usages=0, activated=False)
        elif signal is TriggeringSignal.ROUND_END and self.usages < self.MAX_USAGES:
            return [], replace(self, usages=self.MAX_USAGES)
        return [], self  # pragma: no cover

