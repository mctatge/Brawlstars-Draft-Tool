// Original editorial content for the per-mode draft guides. Kept as data (not JSX) so the
// guide pages stay thin and the copy is easy to review in one place. Mechanics are described
// at a durable level of detail — objectives and draft principles, not patch-specific numbers.
import type { RankedModeName } from "@/lib/reference";

export type GuidePoint = { title: string; body: string };

export type ModeGuide = {
  tagline: string;
  objective: string[];
  draft: GuidePoint[];
  bans: string;
  mapNotes: string;
};

export const MODE_GUIDES: Record<RankedModeName, ModeGuide> = {
  "Gem Grab": {
    tagline: "Control the mine, count to ten, survive the countdown.",
    objective: [
      "A gem mine in the center of the map spits out gems, and the first team to hold ten of them long enough for the victory countdown to finish wins. Gems drop where you die, which makes Gem Grab a control mode wearing an economy mode's clothes: the team that owns the middle of the map owns the gems, and the team that panics at nine gems hands them back.",
      "Most Gem Grab losses aren't lost at the mine — they're lost when the gem carrier gets caught somewhere they had no business being.",
    ],
    draft: [
      {
        title: "Pick a real mid",
        body:
          "One brawler will spend the game contesting the mine, and they have to actually win that job on this map's geometry. On open mines that's usually a long-range damage dealer who controls the sightlines; on walled or bushy mines it's a controller or thrower who makes the area unlivable. A comp where nobody can hold mid is a comp that plays from behind for the whole match.",
      },
      {
        title: "Decide who carries — and protect them",
        body:
          "Gems pile up on one player, and that player becomes the win condition. Good carriers have an escape, sustain, or the range to hold gems from safety. The rest of the draft should include peel for them: healing, knockback, or a body between the carrier and the enemy assassin. Three selfish lane brawlers with no way to protect a ten-gem carrier is a classic losing draft that looks fine on paper.",
      },
      {
        title: "Win one wing",
        body:
          "Gem Grab is usually decided 2-1: hold the mine, win one side lane, and collapse from the winning side. Lane bullies who beat their matchup create constant pressure — every time the enemy sends help to a losing lane, the mine gets cheaper to hold. When you're picking last, look at which lane is weakest and pick the brawler that cracks it.",
      },
    ],
    bans:
      "Ban the brawler that would take the mine for free. On open-mine maps that means the dominant marksmen and artillery of the patch; on walled mines it's the meta throwers, who contest without ever being hittable. If the current meta has an oppressive support that makes enemy carriers unkillable, that's the other ban worth considering.",
    mapNotes:
      "The pool splits into open mines, where sightlines rule and range wins; walled mines, where throwers and tanks thrive because the mine can't be watched from distance; and hybrid maps where the mine is contestable but the side lanes decide who gets to keep it. Check the per-map stats on the draft board before locking a comp that assumes one archetype.",
  },
  "Brawl Ball": {
    tagline: "Two goals wins. The ball doesn't care about your K/D.",
    objective: [
      "Score two goals before the enemy does — or be ahead when the clock runs out. Carrying the ball disables your attacks, so Brawl Ball is a passing and spacing game: the player with the ball is temporarily the weakest player on the field, and the team that positions to receive, screen, and re-tackle wins possession battles.",
      "Kills matter in Brawl Ball only as much as they open the lane to the goal. A team wipe with the ball on your side of the field is worth more than three kills without it.",
    ],
    draft: [
      {
        title: "Bring stop power",
        body:
          "Somebody on the enemy team is going to sprint the ball at your goal — usually a tank. If your draft has no answer (knockback, stun, slow, or a wall of damage at close range), every one of those runs is a coin flip at best. Crowd control converts a scoring run into a turnover, and comps without any concede goals they never had to.",
      },
      {
        title: "Count your goal threats",
        body:
          "Someone has to actually finish. Tanks and assassins who survive the last stretch to the goal, or brawlers whose super carries them (or the ball) through a defense, are your scorers. A comp of three poke brawlers can dominate the field and still fail to convert — control the game and you'll get chances, but only if someone can take them.",
      },
      {
        title: "Respect the walls",
        body:
          "Wall-breaking supers permanently reshape a Brawl Ball map. On tight maps, one wall-breaker turns a stall-fest into open goal lanes; against wall-breakers, a comp that depends on its cover degrades every minute. On already-open maps, skip the demolition and pick the control brawlers who punish dashes across open ground instead.",
      },
    ],
    bans:
      "Tanks headline the ban phase: on tight maps, the meta tanks (or the healer that makes them unstoppable) are the default bans. On open maps, shift bans toward whatever dominates the midfield, because mid control decides who attacks. If one brawler on the current patch both scores and defends at an elite level, that's the ban before any comfort ban.",
    mapNotes:
      "Walled, corridor-style maps play like rugby — tanks, crowd control, and short violent possessions. Open maps play like a control mode where goals happen on power-play advantages. Maps with breakable center walls change character mid-game, which is exactly when wall-breakers and flexible mid brawlers earn their pick.",
  },
  Knockout: {
    tagline: "No respawns. Every death is a round-sized mistake.",
    objective: [
      "Teams of three fight; eliminated players stay down for the rest of the round; win two rounds to take the match. A closing poison ring forces the fight, so you can't hide a lead forever — but you can absolutely lose a round in the first ten seconds by giving up a free pick.",
      "Knockout is the purest positioning mode in ranked. The team that trades health for nothing loses the round before the ring ever closes.",
    ],
    draft: [
      {
        title: "Range is king",
        body:
          "Safe damage rules Knockout. Brawlers who chip from long range force the enemy to choose between eating poke and overextending — both fatal in a no-respawn mode. On open maps, the draft can turn into a straight sniper duel, and showing up without one is how you lose rounds without ever getting to play them.",
      },
      {
        title: "Don't pick feeders",
        body:
          "Every pick needs an answer to the question: how does this brawler avoid dying first? Short-range brawlers with no gap-closer or escape are liabilities into triple range. If you want an aggressive pick, it needs either the mobility to choose its fights or teammates who create the space for it. One early death turns every remaining fight into a 2v3.",
      },
      {
        title: "Plan for the collapse",
        body:
          "When the ring closes, the fight is mandatory and burst wins it. Supers held for the endgame — area denial, multi-hit damage, a well-timed assassination — decide close rounds. Draft at least one brawler who gets stronger when everyone is forced into the same small circle, and play the early round with the endgame in mind: full health at the collapse is a draft-independent superpower.",
      },
    ],
    bans:
      "On open maps, ban the dominant long-range brawlers — the ones that win the poke war outright. On bush-heavy maps, flip it: ban the ambush assassins and close-range monsters who turn every bush into a death sentence. Knockout bans are the most map-dependent in the game; banning an open-map sniper on a bush map is a wasted ban.",
    mapNotes:
      "The pool splits sharply between open, sightline-dominated maps where marksmen rule, and bush-heavy maps where hearing footsteps matters more than aim. Mixed maps usually have one dangerous flank and one safe poke lane — the flank pick decides those. Round length also favors burst over sustained damage everywhere: you don't need to out-damage three enemies, just delete one.",
  },
  "Hot Zone": {
    tagline: "Stand in the circle. No, really — that's the whole mode.",
    objective: [
      "One or more zones sit on the map, and your team's meter fills while you hold them. Fill it first and you win. Nothing about a kill matters except the seconds of uncontested zone time it buys — which is why Hot Zone drafts look different from every other mode's: hit points, healing, and area denial convert directly into progress.",
      "The classic Hot Zone mistake is chasing kills off-zone while the meter quietly decides the game.",
    ],
    draft: [
      {
        title: "Draft brawlers who can stay",
        body:
          "Zone time is the score, so the first slots go to brawlers who can physically remain in or around the circle: tanks with the health to hold ground, controllers who make their half of the zone unenterable, and sustain that turns a losing trade into a stalemate. A glass-cannon trio can win every fight and still lose on the meter because nobody could afford to stand in the paint.",
      },
      {
        title: "Throwers over walls",
        body:
          "When a zone has wall cover, throwers contest it without ever standing in the open — the enemy either eats lobbed damage on the point or leaves. On those maps a thrower is close to mandatory, and the counter-question matters too: if the enemy shows a thrower, someone in your comp needs the mobility to dive them, or the zone becomes rent you pay every second.",
      },
      {
        title: "Keep one fast flex for double zones",
        body:
          "Two-zone maps are rotation puzzles: the winning team stacks one zone, steals seconds on the other, and shifts before the enemy commits. An immobile triple-anchor comp gets outrotated even while winning every direct fight. Draft at least one brawler with real movement who can threaten the far zone and force the enemy to split.",
      },
    ],
    bans:
      "Ban the patch's premier zone-sitters — the tanks and controllers whose sustain makes them nearly impossible to move off the point — and on walled-zone maps, the meta throwers. A useful test: if the answer to 'how do we ever make them leave the zone?' is 'we don't', that's your ban.",
    mapNotes:
      "Single-zone maps are one long teamfight where raw staying power dominates. Double-zone maps reward mobility and map sense as much as combat stats. Zones surrounded by walls are thrower territory; open zones favor whoever wins the mid-range war. The mode has the least comeback potential of the six when the meter gap gets large — early zone control compounds.",
  },
  Heist: {
    tagline: "Crack their safe before they crack yours.",
    objective: [
      "Each team has a safe. Destroy the enemy's, or have dealt more damage to it when time expires. Heist is the tempo mode: defense exists, but every second spent defending is a second not spent on their safe, and the team that dictates which safe is under attack usually wins.",
      "It's also the mode where draft mistakes are least recoverable — if your comp can't threaten the safe, no amount of skill manufactures the missing damage.",
    ],
    draft: [
      {
        title: "Count your safe damage",
        body:
          "The first question of every Heist draft: who actually melts the safe? Brawlers with huge sustained output or supers that dump damage onto a stationary target are the win condition. Be honest in the math — a brawler that's great in fights but mediocre at safe damage is a support pick here, and a comp of three of them loses the race even when it wins the lanes.",
      },
      {
        title: "Assign the defense",
        body:
          "Somebody has to mind the shop. Throwers and area-control brawlers defend the safe while barely leaving the attack, and their lobbed damage punishes divers stacked on your safe. On lane maps, decide at draft time who rotates back — a comp where everyone assumes someone else defends is how safes die to a single unanswered tank.",
      },
      {
        title: "Wall-breakers open the vault",
        body:
          "Most Heist maps hide the safe behind cover, and breaking it changes everything: new attack angles, shorter routes, snipers hitting the safe from range. A wall-breaking super effectively adds a second attacker. Conversely, if your defense depends on those walls, an enemy wall-breaker is a problem you need to answer in the draft, not discover mid-game.",
      },
    ],
    bans:
      "Ban the patch's best safe-melter — the pick that forces the whole enemy draft to be about stopping it — and, on walled maps, the elite defensive throwers who can hold a safe alone. Heist metas tend to concentrate on a few degenerate attackers; if one brawler routinely ends games in the first minute, that ban is not optional.",
    mapNotes:
      "Lane maps make Heist a rotation game: win a lane, trade damage on the safe, get back in time. Open maps become pure races where team-fight control decides who attacks. Water and bridge maps restrict approach routes and inflate the value of brawlers who ignore terrain. The safe's wall cover — and who can remove it — is the single most map-dependent factor in the draft.",
  },
  Bounty: {
    tagline: "Every death hands them stars. Play like yours are worth something.",
    objective: [
      "Kills earn stars, and the team with more stars when the clock ends wins. Dying hands the enemy the stars over your head and resets your own value — so Bounty is the one mode where not dying is literally the objective. A star that spawns mid at the start of the match gives the first fight real stakes.",
      "Leads change how the mode plays: the team ahead can turtle behind its sightlines, and the team behind has to manufacture a pick without gifting one back.",
    ],
    draft: [
      {
        title: "Range wins",
        body:
          "Bounty is the sniper mode. Long sightlines and open ground mean the team that wins the poke war controls the whole map, and short-range brawlers can spend a full match unable to safely enter. Draft range first: marksmen and long-range damage dealers are the spine of nearly every serious Bounty comp, and skimping on it is how you end up spectating your own match.",
      },
      {
        title: "Don't trade down",
        body:
          "A kill for a death is not neutral in Bounty — whoever carried more stars lost the trade. High-value brawlers need discipline and, ideally, an escape or shield to survive their own mistakes. Picks that reliably die once per minute are actively feeding the enemy score no matter how much damage they deal. Survivability tools that look mediocre elsewhere are quietly excellent here.",
      },
      {
        title: "One flanker breaks the stalemate",
        body:
          "When both teams have range and neither blinks, Bounty becomes a staring contest. An assassin or flanker who can slip the sightlines and delete one overextended sniper flips the star ledger in a single play. It's a luxury pick — take it only after the range core is secured, and on maps with the bushes or walls to make the route survivable.",
      },
    ],
    bans:
      "Ban the meta marksmen, nearly always — the brawlers who win the long-range war outright shape the entire match. On the few bushy Bounty maps, spend a ban on the dominant ambusher instead. If the patch has a sniper with both range advantage and an escape, that's the closest thing ranked has to an auto-ban.",
    mapNotes:
      "Most of the pool is open, long-sightline terrain where the range war is the map. A handful of bushier maps invert the mode into ambush chess, and flankers jump from luxury to core there. Center control matters everywhere — the mid star and the best sightlines usually live in the same place, so the first fight tends to set the tone for the next ninety seconds.",
  },
};
