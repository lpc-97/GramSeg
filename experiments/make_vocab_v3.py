#!/usr/bin/env python
# 词表 v3:每行一类(无同义词机制依赖),1k/10k 行,两家 configs 目录同发
import itertools, os
ade = [l.strip() for l in open(os.path.expanduser("~/CorrCLIP/configs/cls_ade20k.txt")) if l.strip()]
assert len(ade) == 150
adjectives = ["red","blue","green","yellow","black","white","small","large","old","new",
  "wooden","metal","plastic","glass","stone","round","square","tall","short","wide",
  "narrow","bright","dark","shiny","rusty","broken","modern","antique","painted","striped",
  "spotted","curved","flat","heavy","light","soft","hard","smooth","rough","clean",
  "dirty","wet","dry","open","closed","empty","full","long","thin","thick","tiny",
  "huge","giant","miniature","folding","electric","manual","digital","vintage","ornate",
  "plain","fancy","leather","cotton","silk","steel","copper","brass","ceramic","marble"]
nouns = ["chair","table","lamp","shelf","cabinet","box","bottle","cup","plate","bowl",
  "basket","bag","hat","coat","shoe","glove","clock","mirror","frame","vase","pot","pan",
  "kettle","toaster","blender","radio","speaker","camera","phone","tablet","laptop",
  "keyboard","mouse","monitor","printer","cable","plug","switch","handle","knob","hinge",
  "bracket","panel","tile","brick","beam","pipe","valve","gauge","engine","wheel","tire",
  "pedal","lever","gear","spring","bolt","screw","nail","hammer","wrench","drill","saw",
  "ladder","bucket","broom","mop","sponge","towel","blanket","pillow","mattress","curtain",
  "rod","hook","rack","stand","tray","jar","lid","cork","straw","napkin","candle","torch",
  "lantern","bell","whistle","horn","drum","flute","violin","trumpet","guitar","piano",
  "bench","stool","crate","barrel","sack","pouch","wallet","purse","briefcase","suitcase",
  "umbrella","cane","helmet","goggles","mask","scarf","belt","buckle","ribbon","thread",
  "needle","button","zipper","collar","cuff","pocket","badge","medal","trophy","statue",
  "figurine","ornament","wreath","garland","banner","flagpole","signpost","mailbox",
  "birdhouse","kennel","cage","aquarium","terrarium","planter","trellis","arbor","gazebo",
  "shed","barn","silo","windmill","watermill","dam","levee","culvert"]
combos = [f"{a} {n}" for a, n in itertools.product(adjectives, nouns)]
for target, fname in [(1000, "cls_vlines1k.txt"), (10000, "cls_vlines10k.txt")]:
    names = ade + combos[: target - 150]
    assert len(names) == target
    for d in ["~/CorrCLIP/configs/", "~/SC-CLIP/configs/"]:
        open(os.path.expanduser(d + fname), "w").write("\n".join(names) + "\n")
    print(fname, len(names))
