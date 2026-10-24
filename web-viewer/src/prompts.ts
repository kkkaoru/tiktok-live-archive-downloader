// Runs with bun (tests) and in the Cloudflare Worker. Every prompt the clip pipeline sends, kept like
// an i18n catalog: each entry has the Japanese `ja` and an English `en` with the same {placeholders},
// and LANGUAGE picks the one that is sent. Japanese is sent: on 191 judged clips the English wording
// saved no tokens and its refitted gate was no better (LOVO AUC 0.752 vs 0.762). Edit both sides
// together; after changing the sent wording, refit the gate's weights on judged clips, since they are
// fitted on the answers to that exact wording. Transcripts are never translated.

export interface Entry {
  ja: string;
  en: string;
}
export interface Levels {
  ja: string[];
  en: string[];
}

const PLACEHOLDER: RegExp = /\{(\w+)\}/g;
/** The language every request is sent in. */
export const LANGUAGE: keyof Entry = "ja";

export const PROMPTS = {
  // What a topic is, said once in every state that asks about topics.
  "topic.rule": {
    ja: "話題とは話している対象・出来事のこと。同じ対象について問いかけ・説明・オチ・反応が続く間は同じ話題。コメントの読み上げ・あいさつ・お礼・配信の進行・別の対象の雑談に移ったら別の話題。音声認識の誤字は気にしない",
    en: "A topic is the subject or event being talked about. While questions, explanations, punchlines and reactions continue about the same subject, the topic stays the same. Moving to reading out comments, greetings, thanks, running the stream or chat about another subject is a new topic. Ignore speech-recognition typos.",
  },
  "topic.strays": {
    ja: "コメントの読み上げ・あいさつ・お礼・配信の進行・別の話題",
    en: "reading out comments, greetings, thanks, running the stream, another topic",
  },
  // Region requests: every question spells out its kind. clef answers a question differently
  // beside other questions, so this exact wording and composition is what the thresholds rest on;
  // a shortened form or a split request shipped fewer good clips.
  "region.state": {
    ja: "ライブ配信の文字起こしの一部。utterancesは発話（timeは配信内の時刻、unitは属する区間の番号）、unitsは区間（excitementは配信内での盛り上がり0〜1）、climaxは山場の区間の番号。{rule}",
    en: "Part of a live stream's transcript. utterances are what was said (time: within the stream, unit: the unit it belongs to), units are the stretches of the stream (excitement: how lively that moment is, 0-1), climax is the number of the peak unit. {rule}",
  },
  "region.boundary": {
    ja: "発話{previous}から発話{utterance}に移るところで、話題・出来事が切り替わっているか（コメント読み・あいさつ・別の雑談への移行も切り替わり）",
    en: "Does the topic or event change from utterance {previous} to utterance {utterance} (moving to reading comments, greetings or other chat counts as a change)?",
  },
  "region.interest": {
    ja: "区間{unit}は、ライブ配信の切り抜き動画の一部として面白いか（会話の山場・笑い・驚き・感情の動きがあるか）",
    en: "Is unit {unit} interesting as part of a short clip from the live stream (a high point of the talk, laughter, surprise, an emotional moment)?",
  },
  "region.opener": {
    ja: "区間{unit}は、疑問・謎・気になる問いかけ・意外な前振りなど、続きを知りたくなる話の起点になっているか",
    en: "Does unit {unit} set up something viewers want to see resolved (a question, a mystery, an intriguing remark, an unexpected setup)?",
  },
  "region.settles": {
    ja: "区間{unit}の最後の発話で、話がオチ・結論・反応の一区切りに達しているか（次の区間が同じ話の続きの途中なら低い）",
    en: "Does the last utterance of unit {unit} bring the story to a punchline, conclusion or reaction (low when the next unit is the middle of the same story)?",
  },
  "region.topic": {
    ja: "区間{unit}は、山場の区間{climax}「{quote}」と同じ話題・出来事について話していて、続けて見ると話がつながるか。{strays}なら低い",
    en: 'Does unit {unit} talk about the same topic or event as the climax unit {climax} "{quote}", so that watching them together connects? Low for {strays}.',
  },
  "region.hook": {
    ja: "区間{unit}をここから見せたとき、最初の{seconds}秒「{opening}」は、視聴者が思わず続きを見たくなる強い引き（疑問・驚き・意外な一言・感情の爆発）か。前置き・あいさつ・相づちだけなら弱い",
    en: 'Shown from here, are the first {seconds} seconds of unit {unit} "{opening}" a strong hook that makes viewers keep watching (a question, surprise, unexpected line, emotional outburst)? Weak when it is only a preamble, greeting or backchannel.',
  },
  // Judge requests: one assembled candidate read alone.
  "draft.state": {
    ja: "ライブ配信から作った切り抜き候補。utterancesは使う発話を時系列順に並べたもの（idは発話の番号、rangeは属する区間の番号）。{rule}",
    en: "A clip candidate cut from a live stream. utterances are the utterances it uses in time order (id: the utterance number, range: the range it belongs to). {rule}",
  },
  "draft.coherent": {
    ja: "この切り抜きの文字起こしは、最初から最後まで同じ一つの話題について話しているか",
    en: "Does this clip's transcript stay on one and the same topic from start to end?",
  },
  "draft.unmixed": {
    ja: "この切り抜きには、主題と関係のない発話（{strays}）が一つも混ざっていないか",
    en: "Is the clip free of any utterance unrelated to its subject ({strays})?",
  },
  "draft.standalone": {
    ja: "配信を見ていない人が、この切り抜きだけで何の話か理解できるか",
    en: "Can someone who did not watch the stream understand what this clip is about from the clip alone?",
  },
  "draft.ending": {
    ja: "この切り抜きの最後の発話で、話がオチ・結論・反応で区切れているか（文や話の途中で終わっていないか）",
    en: "Does the clip's last utterance end the story on a punchline, conclusion or reaction (not in the middle of a sentence or story)?",
  },
  "draft.stray": {
    ja: "発話{utterance}は、この切り抜きの主題と関係のない発話（{strays}）か",
    en: "Is utterance {utterance} unrelated to the clip's subject ({strays})?",
  },
  // The judge's rubric, read by clef and clairic before a clip ships.
  "rubric.state": {
    ja: "ライブ配信の切り抜き動画の文字起こし。区間を示した順に続けて再生する（区間の間は編集でつながる）。音声認識の誤字は気にしない。",
    en: "Transcript of a short clip cut from a live stream. Its ranges play back to back in the order shown (edited together between ranges). Ignore speech-recognition typos.",
  },
  "rubric.story.climax-first": {
    ja: "最初の区間が山場で、続く区間がそこに至る経緯を時系列で見せ、ループ再生で山場に戻る構成。区間の順番は問題にしない",
    en: "The first range is the climax, the following ranges show how it came about in time order, and looping playback returns to the climax. The order of the ranges is not a flaw.",
  },
  "rubric.story.hook-to-climax": {
    ja: "疑問や気になる起点から始まり、最後の区間が山場になる構成。区間の順番は問題にしない",
    en: "It starts from a question or intriguing setup and the last range is the climax. The order of the ranges is not a flaw.",
  },
  "rubric.lean": { ja: "{level}寄り", en: "closer to: {level}" },
  "rubric.topic": {
    ja: "全体が一つの話題・出来事だけを扱っているか",
    en: "Does the whole clip deal with only one topic or event?",
  },
  "rubric.topic.low": {
    ja: "無関係な二つ以上の話題が混在する",
    en: "two or more unrelated topics are mixed",
  },
  "rubric.topic.mid": {
    ja: "主題はあるが無関係な発話が少し混じる",
    en: "there is a main topic but a few unrelated utterances",
  },
  "rubric.topic.high": { ja: "最初から最後まで同じ話題", en: "the same topic from start to end" },
  "rubric.standalone": {
    ja: "配信を見ていない人がこの切り抜きだけで何の話か理解できるか",
    en: "Can someone who did not watch the stream understand what this clip is about from the clip alone?",
  },
  "rubric.standalone.low": { ja: "何の話か分からない", en: "impossible to tell what it is about" },
  "rubric.standalone.mid": { ja: "少し分かりにくい", en: "somewhat hard to follow" },
  "rubric.standalone.high": { ja: "前提がすべて含まれる", en: "every premise is included" },
  "rubric.hook": {
    ja: "最初の1〜2発話が続きを見たくさせるか",
    en: "Do the first one or two utterances make viewers want to keep watching?",
  },
  "rubric.hook.low": {
    ja: "あいさつ・相づち・前置きだけ",
    en: "only greetings, backchannels or a preamble",
  },
  "rubric.hook.mid": { ja: "普通", en: "ordinary" },
  "rubric.hook.high": {
    ja: "強い疑問・驚き・感情の爆発",
    en: "a strong question, surprise or emotional outburst",
  },
  "rubric.ending": {
    ja: "最後が話のオチ・結論・反応で区切れているか",
    en: "Does the clip end on the story's punchline, conclusion or reaction?",
  },
  "rubric.ending.low": {
    ja: "文や話の途中で切れる",
    en: "cut off in the middle of a sentence or story",
  },
  "rubric.ending.mid": { ja: "一応区切れる", en: "ends acceptably" },
  "rubric.ending.high": { ja: "きれいにオチる", en: "lands cleanly" },
  "rubric.flow": {
    ja: "発話のつながりが自然で話が飛ばないか。文の途中から始まる・途中で終わるなら低い",
    en: "Do the utterances connect naturally without jumps? Low when it starts or ends in the middle of a sentence.",
  },
  "rubric.flow.low": { ja: "支離滅裂", en: "incoherent" },
  "rubric.flow.mid": { ja: "少し飛ぶ", en: "jumps a little" },
  "rubric.flow.high": { ja: "自然", en: "natural" },
  "rubric.good": {
    ja: "この切り抜きは、一つの話題だけを扱い、配信を見ていない人にも分かり、最後がオチ・結論・反応で区切れ、発話が自然につながっているか",
    en: "Does this clip deal with a single topic, make sense to someone who did not watch the stream, end on a punchline, conclusion or reaction, and connect its utterances naturally?",
  },
  "rubric.premise": {
    ja: "話の前提（誰が・何が・何の話か）が切り抜きの中で説明されているか",
    en: "Does the clip itself explain the premise (who, what, what it is about)?",
  },
  "rubric.startsMid": {
    ja: "最初の発話は、文や話の途中から始まっているか",
    en: "Does the first utterance start in the middle of a sentence or story?",
  },
  "rubric.endsMid": {
    ja: "最後の発話は、文や話の途中で終わっているか",
    en: "Does the last utterance end in the middle of a sentence or story?",
  },
  "rubric.stray": {
    ja: "コメントの読み上げ・あいさつ・お礼・配信の進行・別の話題の発話が一つでも混ざっているか",
    en: "Is even one utterance of reading out comments, greetings, thanks, running the stream or another topic mixed in?",
  },
  // clairic reads one line per utterance, so the context and story travel with every field.
  "clairic.context": {
    ja: "文はライブ配信の切り抜き動画の文字起こしで、[区間n] の順に続けて再生する（区間の間は編集でつながる）。音声認識の誤字は気にしない。",
    en: "The text is the transcript of a short clip cut from a live stream; it plays [range n] after [range n] back to back (edited together between ranges). Ignore speech-recognition typos. ",
  },
  "clairic.line": { ja: "[区間{range}] {text}", en: "[range {range}] {text}" },
  "clairic.story.climax-first": {
    ja: "構成: 最初の区間が山場で、続く区間がそこに至る経緯を時系列で見せ、ループ再生で山場に戻る構成。区間の順番は問題にしない。",
    en: "Story: the first range is the climax, the following ranges show how it came about in time order, and looping playback returns to the climax. The order of the ranges is not a flaw. ",
  },
  "clairic.story.hook-to-climax": {
    ja: "構成: 疑問や気になる起点から始まり、最後の区間が山場になる構成。区間の順番は問題にしない。",
    en: "Story: it starts from a question or intriguing setup and the last range is the climax. The order of the ranges is not a flaw. ",
  },
  "clairic.topic": {
    ja: "全体が一つの話題・出来事だけを扱っているか。1=無関係な二つ以上の話題が混在、3=主題はあるが無関係な発話が少し混じる、5=最初から最後まで同じ話題",
    en: "Does the whole clip deal with only one topic or event? 1 = two or more unrelated topics mixed, 3 = a main topic with a few unrelated utterances, 5 = the same topic from start to end",
  },
  "clairic.standalone": {
    ja: "配信を見ていない人がこの切り抜きだけで何の話か理解できるか。1=分からない、3=少し分かりにくい、5=前提がすべて含まれる",
    en: "Can someone who did not watch the stream understand what this clip is about from the clip alone? 1 = no, 3 = somewhat hard to follow, 5 = every premise is included",
  },
  "clairic.hook": {
    ja: "最初の1〜2発話が続きを見たくさせるか。1=あいさつ・相づち・前置きだけ、3=普通、5=強い疑問・驚き・感情の爆発",
    en: "Do the first one or two utterances make viewers want to keep watching? 1 = only greetings, backchannels or a preamble, 3 = ordinary, 5 = a strong question, surprise or emotional outburst",
  },
  "clairic.ending": {
    ja: "最後が話のオチ・結論・反応で区切れているか。1=文や話の途中で切れる、3=一応区切れる、5=きれいにオチる",
    en: "Does the clip end on the story's punchline, conclusion or reaction? 1 = cut off mid-sentence or mid-story, 3 = ends acceptably, 5 = lands cleanly",
  },
  "clairic.flow": {
    ja: "発話のつながりが自然で話が飛ばないか。文の途中から始まる・途中で終わるなら低い。1=支離滅裂、3=少し飛ぶ、5=自然",
    en: "Do the utterances connect naturally without jumps? Low when it starts or ends mid-sentence. 1 = incoherent, 3 = jumps a little, 5 = natural",
  },
  "clairic.punchline": {
    ja: "話のオチ・結論・山場の反応になっている発話。無ければ null",
    en: "The utterance that is the story's punchline, conclusion or climactic reaction; null if there is none",
  },
  "clairic.premise": {
    ja: "話の前提（誰が・何が・何の話か）を説明している発話。無ければ null",
    en: "The utterance that explains the premise (who, what, what it is about); null if there is none",
  },
  // The editor titles the clips that ship; its answers stay Japanese for the viewers.
  "editor.system": {
    ja: [
      "あなたはライブ配信の切り抜き動画にタイトルと要約を付ける担当者です。候補ごとに、配信のいくつかの区間を示した順に続けて再生します。",
      "文字起こし（再生順、音声認識の誤字は無視）だけを根拠にします。",
      "titleは最初の3秒の引きを生かして続きが気になる25字以内、summaryは120字以内の日本語。文字起こしにない出来事や別の話題を作らないこと。",
      'すべての候補について、JSONだけを返す: {"clips": [{"id": 候補の番号, "title": "...", "summary": "..."}]}',
    ].join("\n"),
    en: [
      "You title and summarize short clips cut from a live stream. Each candidate plays several ranges of the stream back to back in the order shown.",
      "Rely only on the transcript (in playback order; ignore speech-recognition typos).",
      "Write both in Japanese: title within 25 characters, using the hook of the first 3 seconds so viewers want to see more; summary within 120 characters. Do not invent events or topics that are not in the transcript.",
      'For every candidate, return only JSON: {"clips": [{"id": candidate number, "title": "...", "summary": "..."}]}',
    ].join("\n"),
  },
  "editor.candidate": {
    ja: "## 候補{number}（{seconds}秒） 構成: {story}",
    en: "## Candidate {number} ({seconds} s) Story: {story}",
  },
  "editor.range": { ja: "[区間{range}]", en: "[range {range}]" },
  "editor.story.climax-first": {
    ja: "最初の区間がこの話の山場（一番の盛り上がり）で、続く区間はそこに至るまでの経緯を時系列で見せる構成",
    en: "the first range is the story's climax (its biggest moment), and the following ranges show how it came about in time order",
  },
  "editor.story.hook-to-climax": {
    ja: "最初の区間で疑問・気になる起点を見せ、区間が進むにつれて盛り上がり、最後の区間が山場になる構成",
    en: "the first range shows a question or intriguing setup, the ranges build up, and the last range is the climax",
  },
} satisfies Record<string, Entry>;

export const LEVELS = {
  interest: {
    ja: ["退屈", "やや退屈", "普通", "面白い", "とても面白い"],
    en: ["boring", "a little boring", "ordinary", "interesting", "very interesting"],
  },
  hook: {
    ja: ["引きがない", "弱い", "普通", "強い", "非常に強い"],
    en: ["no hook", "weak", "ordinary", "strong", "very strong"],
  },
} satisfies Record<string, Levels>;

export type PromptKey = keyof typeof PROMPTS;

/** The {placeholders} of a template, sorted. */
export const placeholders = (template: string): string[] =>
  [...template.matchAll(PLACEHOLDER)].flatMap((m) => m.slice(1, 2)).sort();

/** The sent text of `key` with its placeholders filled; a missing value is a bug, so it throws. */
export function prompt(key: PromptKey, values?: Record<string, string | number>): string {
  return PROMPTS[key][LANGUAGE].replace(PLACEHOLDER, (_, name: string) => {
    const value = values?.[name];
    if (value === undefined) throw new Error(`prompt ${key} needs {${name}}`);
    return String(value);
  });
}
