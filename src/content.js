// 站点文案集中于此。信息均摘自资料库源文件：
// 简历/微信常用简介.md · 简历/董晟渤简历 2026.docx · 简历/2026届学生案例.md
// 简历/高考数学培优课程方案_202607.pdf · 讲义/ · 出版物/

// 图片统一放 public/images/ 下，文件按下面这些名字命名即可；缺图时页面会显示占位框。
// 注意：教师形象照（hero）未放入前不会显示任何占位，放图后自动出现。
export const images = {
  hero: './images/hero-photo.jpg', // 教师形象照 / 北大认证照片（竖版 3:4），未放图前隐藏
  outcomes: './images/student.webp', // 学员出分的微信聊天截图（原“喜报拼图”位改为该图）
  qrWechat: './images/qr-wechat-square.png', // 个人微信二维码（已修成规整方图）
  qrGzh: './images/qr-gzh.jpg', // 微信公众号二维码（方形）
  coverDaoshu: './images/cover-daoshu.jpg', // 《高考导数探秘》封面
  coverYuanzhui: './images/cover-yuanzhui.jpg', // 《高考圆锥曲线探秘》封面
  coverGailv: './images/cover-gailv.jpg', // 《高考概率统计压轴突破》讲义封面（对应未出版书稿《高考概率统计探秘》）
}

export const profile = {
  name: '董晟渤',
  latin: 'Dylaaan',
  tagline: '高考数学培优与强基',
  position: '北京大学数学科学学院 博士在读 · 高考数学与强基拔尖培优',
  thesis: '科班数学出身，以正统的数学视角，把题目背后的数学本质讲清楚，帮助学生构建自己的解题体系。',
  thesisNote:
    '优秀的学生更应该把基本功做扎实：概念清楚、方法成体系、贴近高考考法，而不是背“套路”或堆砌偏题怪题。我始终重视高考真题——真题有自己独特的味道，值得反复研读体会；教学中我力求讲清方法背后的数学本质，帮学生建立可迁移的解题结构。',
  quickFacts: [
    { label: '在读学位', value: '北大数学 概率统计方向 博士' },
    { label: '著作体系', value: '高考压轴“三书” + 配套讲义' },
    {
      label: '线上教研',
      value: '知乎数学话题优秀答主 · 17W+ 关注',
      url: 'https://www.zhihu.com/people/dylan-dong-233',
    },
  ],
}

export const navLinks = [
  { id: 'academic', label: '教研背景' },
  { id: 'publications', label: '出版专著' },
  { id: 'outcomes', label: '教学成果' },
  { id: 'curriculum', label: '课程体系' },
  { id: 'contact', label: '咨询' },
]

export const education = [
  {
    org: '北京大学',
    period: '2023 — 至今',
    role: '数学科学学院 · 概率论与数理统计方向 博士在读',
    note: '在校担任《高等数学》《概率论》等课程助教，研究方向为概率论与随机过程。',
  },
  {
    org: '北京大学',
    period: '2023 — 至今',
    role: '教育学院 “国优计划” 教育硕士在读',
    note: '教育部“国家优秀中小学教师培养计划”首批成员，免试认定中小学教师资格。',
  },
  {
    org: '西安交通大学',
    period: '2019 — 2023',
    role: '数学与统计学院 · 统计学学士',
    note: '本科以专业第一名的成绩保送北京大学。',
  },
]

export const media = {
  title: '教研传播 · Dylaaan',
  platform: '知乎数学话题优秀答主 · 2025 年度新知答主',
  account: '账号 Dylaaan · 全网关注 17W+',
  zhihu: 'https://www.zhihu.com/people/dylan-dong-233',
  note: '2018 年，还在读高中的我在知乎写下第一篇数学文章；多年后成为北大数院概率统计方向研究生，接触了更高等、更现代的数学，回头再看高考问题，理解自然更深了一层——这也是我写作与授课的底色。',
}

export const experience = {
  items: [
    {
      title: '公立学校 · 教科院培优教练',
      body: '曾在北京二十中学、四川绵阳南山中学、四川南充高中、湖北天门中学、河北石家庄一中、陕西咸阳实验中学等校，以及山东潍坊市教科院主讲高考培优、强基与教研课程。',
    },
    {
      title: '教育机构 · 培优课程合作',
      body: '与北大金秋教育、北京学为教育、培尖教育、学杰燕园、中佳九学等机构合作开发并讲授高考数学培优课程。',
    },
    {
      title: '线上专题课 · 公开讲座',
      body: '开设《高考概率统计压轴突破》《考前专题：组合与概率统计选讲》、导数高阶技巧等线上专题课与备考讲座，累计线上学员数百人，覆盖全国卷多个省份。',
    },
    {
      title: '校内教学 · 助教',
      body: '在北京大学担任《高等数学》《概率论》等课程助教，并就读“国优计划”，具备中学教师培养与从业资质。',
    },
  ],
}

export const books = [
  {
    id: 'daoshu',
    status: '已出版',
    statusClass: 'bg-emerald-50 text-emerald-800 border-emerald-200',
    title: '高考导数探秘',
    subtitle: '解题技巧与策略 · 人民邮电出版社',
    coverNote: '[配图：出版物/导数排版文件/封面.pdf 渲染图]',
    img: images.coverDaoshu,
    desc: '把导数压轴题讲成体系：从函数基本功出发，依次打通分类讨论、函数不等式、双变量、极值点偏移与隐零点等主线问题——贴近高考、不讲偏题怪题。',
    points: ['分类讨论与分离参数', '函数不等式', '双变量 / 多变量问题', '极值点偏移', '隐零点问题'],
  },
  {
    id: 'yuanzhui',
    status: '已出版',
    statusClass: 'bg-emerald-50 text-emerald-800 border-emerald-200',
    title: '高考圆锥曲线探秘',
    subtitle: '从体系到技巧 · 人民邮电出版社',
    coverNote: '[配图：出版物/圆锥曲线排版文件/封面.pdf 渲染图]',
    img: images.coverYuanzhui,
    desc: '回归圆锥曲线的几何本质，以齐次化联立、参数方程与压缩变换等方法精简计算，让解析几何“算得少、想得清”。',
    points: ['几何本质与统一定义', '齐次化联立', '参数方程与压缩变换', '复杂计算优化'],
  },
  {
    id: 'gailv',
    status: '已成体系',
    statusClass: 'bg-slate-100 text-slate-700 border-slate-300',
    title: '高考概率统计探秘',
    subtitle: '从本质建立概率统计的知识体系',
    courseLink: {
      name: '《高考概率统计压轴突破》',
      href: 'https://mp.weixin.qq.com/s/o-Yl8fE7IteFNXNA0nGhUA',
    },
    coverNote: '[配图：《高考概率统计压轴突破》课程讲义封面]',
    img: images.coverGailv,
    desc: '应对概率统计与新高考创新压轴：从离散概率的公理化出发，贯穿期望递推、随机过程，延伸至一元回归与统计推断。',
    points: ['计数原理与概率', '期望与随机变量的递推', '一元线性回归模型', '新高考创新压轴延伸'],
  },
]

export const handoutNote = {
  title: '自编讲义 · 例题精讲，随课同步',
  desc: '每一讲都配有排版精良的自编讲义：例题先把方法讲透，练习留给你独立思考，再逐步给出解析。题组以高考真题为骨架、适当融入强基题，不收录偏题怪题。',
  flow: '讲义沉淀成书：三本“探秘”专著正是由对应的专题讲义体系升级而来。',
  tags: ['例题精讲 + 练习解析', '高考真题 + 强基题组', '出版级排版'],
}

export const voice = {
  origin:
    '这套体系的起点，是 2018 年我在知乎写下的高中文章；它的成熟，来自北大数学训练带来的更高视角，也来自一届届学生的真实反馈。无论是书还是讲义，都不是题目与答案的堆砌，而是我对概念与题目的理解。',
  principles: [
    '不教背“套路”，教思维方法——考场上灵活“见招拆招”，用合适的工具解题。',
    '对知识点、技巧与方法有本质的理解，才能建立体系，而不是掉进“背二级结论”的陷阱。',
    '以高考真题与强基真题为纲，拒绝偏题、怪题与“野题”，把时间花在刀刃上。',
  ],
  signature: '—— 董晟渤（Dylaaan）· 摘自《高考导数探秘》《高考圆锥曲线探秘》前言',
}

export const outcomes = {
  cohort: '2026 届',
  sourceNote:
    '数据引自作者于高考出分后公开发布的《2026 届学生案例》汇报，并已于多所合作学校、机构处存档。',
  stats: [
    { value: '8 位', label: '学员达到清北裸分 / 强基入围位次' },
    { value: '7 位', label: '学员高考数学单科 140+（最高 149）' },
    { value: '11 位', label: '学员总分 650+，覆盖 8 个省份' },
  ],
  rankHighlights: [
    { rank: '重庆 前 30', score: '数学 149 · 总分 699' },
    { rank: '福建 前 50', score: '数学 142 · 总分 693' },
    { rank: '福建 前 50', score: '数学 140 · 总分 692' },
    { rank: '山东 前 70', score: '数学 140 · 总分 696' },
    { rank: '山东 前 160', score: '数学 136 · 总分 691' },
  ],
}

export const outcomeRows = [
  { math: '149', total: '699', rank: '重庆 前 30 名', mode: '线上专题' },
  { math: '142', total: '693', rank: '福建 前 50 名', mode: '入校课' },
  { math: '140', total: '692', rank: '福建 前 50 名', mode: '入校课' },
  { math: '133', total: '686', rank: '福建 前 50 名', mode: '入校课' },
  { math: '140', total: '696', rank: '山东 前 70 名', mode: '入校课 + 一对一' },
  { math: '134', total: '694', rank: '山东 前 90 名', mode: '入校课 + 一对一' },
  { math: '136', total: '691', rank: '山东 前 160 名', mode: '入校课 + 一对一' },
  { math: '122', total: '652', rank: '云南（历史类）前 90 名', mode: '线上专题' },
]

export const curriculum = {
  intro:
    '三大专题的讲法一以贯之：回到概念与原理，拒绝堆砌二级结论，在通性通法与严格论证中，建立从“会做”到“会想”的进阶路径。',
  suitable: {
    title: '适合对象',
    body: '平时数学成绩在 110 分以上，或目标在 130+，希望系统攻克第 17–19 题压轴与强基拔尖题的学生（高一至高三均可）。',
    note: '课程起点依诊断结果定制，重在查漏与结构重建，而非重复刷题。',
  },
  tracks: [
    {
      icon: 'FunctionSquare',
      title: '函数与导数专题',
      eng: 'Functions & Derivatives',
      body: '以分类讨论、分离参数等基本功入手，打通函数不等式、多变量、极值点偏移与隐零点等主线，建立适配全国卷命题风格的导数解题体系；贴近高考，不搞偏题怪题。',
      tags: ['体系化 · 贴近高考', '重视基本功', '一题多解与通法', '对接《高考导数探秘》'],
    },
    {
      icon: 'Grid3x3',
      title: '解析几何专题',
      eng: 'Analytic Geometry',
      body: '从椭圆、双曲线、抛物线的定义出发，系统梳理联立、齐次化联立与“不联立”等计算路径，把原理讲清楚、把该算的算明白，尽量减少计算量。',
      tags: ['几何本质', '齐次化联立', '参数与代换技巧', '对接《高考圆锥曲线探秘》'],
    },
    {
      icon: 'ChartSpline',
      title: '概率统计与创新题',
      eng: 'Probability & Innovation',
      body: '系统梳理概率统计真题，讲清对称、分解、递推等真正高频的技巧，覆盖随机变量、随机过程等新高考创新压轴，并延伸到统计推断。',
      tags: ['随机变量与递推', '强基交叉题型', '对接《高考概率统计探秘》'],
    },
  ],
  modes: [
    {
      title: '线上专题精讲班',
      body: '直播或录播课形式，报名人数一般不设上限；随堂发放自编讲义与练习，课后保留答疑。',
    },
    {
      title: '定制化小班',
      body: '面向希望获得更高关注度的学员：按需定制组班规模、授课内容与节奏，一对一亦可协商。',
    },
    {
      title: '入校讲座 / 驻校培优',
      body: '面向重点高中的压轴专题与强基课程，可对接学校或教科院开展培优及教研合作。',
    },
  ],
}

export const contact = {
  wechat: 'Dylan_PKU',
  note: '为保证交流效率，添加时请务必备注：【年级 + 省份 + 当前数学平时分数】；如需报名概率统计线上课程，请再加备注【概率统计】。',
  qrNote: '[配图：个人微信二维码图片]',
  gzh: {
    name: '微信公众号',
    note: '数学文章与课程通知，可扫码关注。',
    qrNote: '[配图：微信公众号二维码]',
  },
  responseTime: '通常在 1 个工作日内回复，请耐心等待。',
}
