from ast import Return
from docx import Document
import interface
import tk
import docx


def get_text(indicate):
    raz = ["1. ПРЕДМЕТ ДОГОВОРА",
           f"1.1 Заказчик поручает, а Исполнитель принимает на себя обязательства оказание (выполнение) следующих услуг (работ): [_predmetDog_].",
           "1.2. Услуга (работа) приобретается заказчиком для собственного производства и (или) потребления.",
           f"1.3. Источник финансирования – [_fin_]",
           "",
           "2. ПОРЯДОК РАСЧЕТОВ И ЦЕНА УСЛУГИ",
           "2.1. Цены на услуги (работы) определяются Исполнителем и согласовываются с Заказчиком в счёт – протоколе согласования отпускной цены (Приложения №1), являющейся неотъемлемой частью настоящего договора",
           "2.2. Оплата услуги – по факту оказания (выполнения) в течение 10 (десяти) банковских дней.",
           "2.3. Расчеты осуществляются в безналичной форме в белорусских рублях со счетов Заказчика.",
           "",
           "3. СРОК И ПОРЯДОК ОКАЗАНИЯ УСЛУГИ",
           "3.1. Срок оказания услуг (работы) согласовываются сторонами и указываются в счёт – протоколе согласования отпускной цены (Приложения №1).",
           "3.2. Услуга (работа) считается оказанной после подписания акта выполненных работ (акта сдачи-приемки работ) «Заказчиком» или его полномочным представителем.",
           "",
           "4. КАЧЕСТВО УСЛУГИ",
           "4.1. Услуга (работа) должна быть оказана (выполнена) качественно в соответствии с ТНПА и(или) нормативно-технической документацией на оказание данного вида услуги (работы).",
           "4.2. Гарантийный срок исчисляется с момента выполнения работ в соответствии с требованиями нормативной и (или) технической документации на выполняемую работу, но не менее 12 месяцев.",
           "",
           "5. САНКЦИИ, ОТВЕТСТВЕННОСТЬ СТОРОН.",
           "5.1. За неисполнение либо не надлежащее исполнение условий договора стороны несут ответственность в соответствии с действующим законодательством Республики Беларусь.",
           "",
           "6. ФОРС-МАЖОР",
           "6.1. Стороны освобождаются от ответственности за частичное или полное неисполнение условий договора, если оно произошло по обстоятельствам непреодолимой силы: пожара, стихийных бедствий, военных действий любого характера, которые сторона не могла предвидеть или предотвратить. Сторона, ссылающаяся на такие обстоятельства, обязана информировать другую сторону не позднее 5 дней с момента их наступления. Наступление вышеуказанных обстоятельств должно быть подтверждено документом, выданным уполномоченным органом.",
           "6.2. При возникновении обстоятельств непреодолимой силы срок выполнения обязательств по настоящему договору отодвигается соразмерно времени, в течение которого действуют такие обстоятельства.",
           "",
           "7. ПРОЧИЕ УСЛОВИЯ",
           "7.1. Изменение условий договора возможно только по обоюдному соглашению сторон, оформленному в письменном виде.",
           "7.2. Споры и разногласия, возникающие при заключении и исполнении настоящего Договора, решаются путем переговоров. При недостижении согласия сторона, чьи права или законные интересы нарушены, обязана предъявить другой стороне претензию. Получатель претензии в 7-дневный срок со дня ее получения письменно уведомляет заявителя претензии о результатах рассмотрения претензии. Неполучение ответа на претензию в 7-дневный срок не препятствует обращению стороны, предъявившей претензию, в Экономический суд Гомельской области.",
           "7.3. Настоящий Договор вступает в силу с момента подписания обеими сторонами и действует до полного исполнения обязательств.",
           "7.4. Настоящий Договор подписан в двух экземплярах, имеющих одинаковую юридическую силу, по одному для каждой из сторон.",
           "7.5. В остальном, что не урегулировано настоящим договором, стороны руководствуются действующим законодательством Республики Беларусь.",
           "7.6. Настоящий договор, дополнительные соглашения, изменения, приложения, спецификации, а также вся относящаяся к нему переписка могут быть переданы посредством факсимильной связи или электронной почтой с электронной цифровой подписью. Факты заключения Сторонами Договора, дополнительных соглашений и совершения изменений считаются подтвержденными при условии последующего обмена Сторонами оригиналами факсимильных или электронных сообщений на бумажном носителе.",
           "",
           "8. ЮРИДИЧЕСКИЕ АДРЕСА СТОРОН"]
    dolg = ["1. ПРЕДМЕТ ДОГОВОРА",
            "1.1.	Заказчик поручает, а Подрядчик принимает на себя обязательства по выполнению следующих видов работ, далее Работ:",
            "1.1.1.	первоначальное обследование подъемников (далее – «подъемное оборудование» проводится 1 раз перед началом обслуживания;",
            "1.1.2.	техническое обслуживание (ТО-1) проводится 1 раз в месяц с 10 по 30 числа текущего месяца. Не проводится в месяце проведения ТО-2;",
            "1.1.3.	техническое обслуживание (ТО-2) проводится 1 раз в год с 10 по 30 декабря;",
            "1.1.4.	ежегодное техническое освидетельствование проводится 1 раз год с 10 по 30 декабря после проведения ТО-2;",
            "1.1.5.	работы по ремонту и замене составных частей подъемного оборудования, не входящие в состав технического обслуживания, работы по ремонту подъемного оборудования в случаях хищения, умышленной порчи подъемного оборудования",
            "1.2.	Подъемное оборудование расположено на объекте: г. Гомель, ул. Чечерская, 60",
            "1.3.	Работа включает в себя: проведение планового технического обслуживания подъемного оборудования, состоящего в проверке работоспособности Устройств, их внешнем осмотре, проведении профилактических работ планово-предупредительного характера для поддержания Устройств в работоспособном и исправном состоянии в соответствии с действующими техническими нормативно-правовыми актами, технологическими картами и технической документацией предприятий-изготовителей, техническое освидетельствование.",
            "",
            "2. ОБЯЗАННОСТИ СТОРОН",
            "2.1.	Подрядчик обязуется обеспечить содержание подъемного оборудования в технически исправном состоянии. В этих целях:",
            "2.1.1.	Назначить лиц, ответственных за проведение технического обслуживания и ремонт подъемного оборудования, с правом приемки и подписания актов выполненных работ по договору.",
            "2.1.2.	Перед принятием подъемного оборудования на техническое обслуживание провести обследование на предмет возможности проведения технического освидетельствования.",
            "2.1.3.	Организовать проведение технического обслуживания и ремонта подъемного оборудования.",
            "2.1.4.	Выполнять ремонт подъемного оборудования, вышедшей из строя в результате хищения или умышленной порчи оборудования, а также в случаях нарушения Заказчиком правил эксплуатации подъемников, в соответствии с дефектным актом.",
            "2.1.5.	Организовать обучение и периодическую проверку знаний персонала, осуществляющего обслуживание подъемного оборудования.",
            "2.1.6.	Проводить ежегодное техническое освидетельствование подъемного оборудования.",
            "2.1.7.	Представлять интересы Заказчика (владельца подъемного оборудования), на заводе-изготовителе подъемного оборудования, по вопросам его гарантийного ремонта и качества изготовления.",
            "2.1.8.	Своевременно вносить в паспорт соответствующие записи, касающиеся технического состояния и ремонта подъемного оборудования.",
            "2.1.9.	Провести разъяснительную работу с представителем Заказчика по Правилам безопасного пользования подъемного оборудования, по бережному отношению к подъемному оборудованию.",
            "2.1.10.	Ставить в известность Заказчика о фактах хищений и умышленной порчи оборудования.",
            "2.1.11.	Приостанавливать работу подъемного оборудования при наличии неисправностей, влияющих на ее безопасную эксплуатацию.",
            "2.1.12.	Своевременно уведомлять Заказчика о необходимости замены отдельных деталей, узлов и механизмов, дальнейшая эксплуатация которых не обеспечивает безопасную и бесперебойную работу подъемного оборудования, а также информировать Заказчика об изменениях требований к эксплуатации подъемного оборудования.",
            "2.1.13.	Возвращать Заказчику все демонтированные узлы и детали, в том числе содержащие драгоценные металлы, по акту приемки передачи. Акт со стороны Подрядчика подписывается лицом ответственным за исправное состояние, техническое обслуживанием ремонт подъемного оборудования.",
            "2.1.14.	Обеспечить выполнение ответственными специалистами и обслуживающим персоналом требований правил, должностных и производственных инструкции.",
            "",
            "2.2. ЗАКАЗЧИК ОБЯЗУЕТСЯ:",
            "2.2.1.	Предоставлять по требованию Подрядчика паспорт и техническую документацию на подъемное оборудование для внесения необходимой информации.",
            "2.2.2.	Обеспечить бесперебойную подачу электроэнергии на вводные устройства подъемного оборудования. Содержать в исправном состоянии предохранительные устройства и электропроводку до вводного устройства (Граница раздела эксплуатационной ответственности проходит по верхним контактам автомата).",
            "2.2.3.	Обеспечить бесперебойную подачу электроэнергии, обеспечить освещение посадочных площадок.",
            "2.2.4.	Обеспечить сохранность находящегося на балансе у Заказчика подъемного оборудования. Незамедлительно информировать местные органы внутренних дел о фактах хищений или умышленной порчи подъемного оборудования.",
            "В случае невыполнения пункта 2.2.4 Подрядчик не несет ответственности за исправное техническое состояние оборудования подъемного оборудования и обязан вывести его из эксплуатации.",
            "2.2.5.	Обеспечить санитарную уборку подъемного оборудования и погрузочных площадок.",
            "2.2.6.	Выполнять в установленные сроки предписания Подрядчика в части, касающейся Заказчика.",
            "2.2.7.	Делегировать права Подрядчику представлять интересы Заказчика (владельца подъемного оборудования), на заводе-изготовителе подъемного оборудования, по вопросам его гарантийного ремонта и качества изготовления.",
            "2.2.8.	Участвовать в проведении ежегодного технического освидетельствования подъемного оборудования, в ее дефектации перед ремонтом, а также в составлении акта о хищении и умышленной порче подъемного оборудования и определении размера нанесенного при этом ущерба.",
            "2.2.9.	Обеспечить подъемное оборудование Правилами безопасного пользования подъемного оборудования и табличками с указанием номеров телефонов аварийной службы.",
            "2.2.10.	Проводить разъяснительную работу с пользователями по безопасному пользованию подъемным оборудованием, бережному отношению к подъемному оборудованию.",
            "2.2.11.	Рассматривать и подписывать в течение трех суток по представлению Подрядчика акты выполненных работ, а в случае несогласия дать письменный аргументированный отказ.",
            "2.2.12.	Информировать в 5-дневный срок Подрядчика об изменениях почтовых и банковских реквизитов.",
            "2.2.13.	Определить порядок хранения и учета выдачи ключей от устройств управления подъемным оборудованием.",
            "2.2.14.	Обеспечить работу аварийной службы в целях оперативного устранения возникших сбоев в работе подъемного оборудования.",
            "2.2.15.	Назначить лиц, ответственных за безопасную эксплуатацию и исправное состояние подъемного оборудования, с правом приемки и подписания актов выполненных работ по договору.",
            "2.2.16.	Принять от Подрядчика все демонтированные узлы и детали, в том числе содержащие драгоценные металлы, по акту приемки передачи. Акт со стороны Заказчика подписывается лицом, ответственным за безопасную эксплуатацию и исправное состояние подъемного оборудования.",
            "",
            "3. ЦЕНЫ И ПОРЯДОК РАСЧЕТОВ",
            "3.1.	Стоимость на работы устанавливаются в белорусских рублях.",
            "3.2.	Стоимость Работ на 2023 год по настоящему договору составляет 1 227,93 (Одна тысяча двести двадцать семь белорусский рубль 93 копейки), без НДС, ",
            "3.3.	Стоимость на Работы определяется Подрядчиком и согласовывается с Заказчиком в Спецификации/Протоколе согласования договорной цены (Приложение № 1).",
            "3.4.	Приложения к настоящему Договору составляют его неотъемлемую часть.",
            "3.5.	Источник финансирования - [_fin_]",
            "3.6.	Оплата выполненных Работ производится Заказчиком путем перечисления денежных средств на расчетный счет Подрядчика в течение 10-ти банковских дней с момента подписания акта выполненных работ.",
            "3.7.	В цену технического обслуживания подъемного оборудования включена стоимость расходных материалов.",
            "3.8.	В цену технического обслуживания подъемного оборудования не включена стоимость комплектующих и узлов, дефект которых выявлен при техническом обслуживании и оплачивается Заказчиком дополнительно на основании дефектного акта.",
            "3.9.	Работы капитального характера по ремонту и замене составных частей подъемного оборудования, не входящие в состав технического обслуживания, работы по ремонту подъемного оборудования в случаях хищения, умышленной порчи подъемного оборудования и нарушения Заказчиком правил эксплуатации, оплачиваются Заказчиком отдельно по предъявленным калькуляциям и актам выполненных работ. Стоимость материалов в данном случае Заказчик оплачивает предварительно, а окончательный расчет производится в 15-дневный срок после подписания актов выполненных работ.",
            "3.10.	При простое подъемного оборудования свыше одного месяца по причине, не зависящей от Подрядчика, перед их запуском Подрядчик проводит обследование на предмет возможности проведения технического освидетельствования и техническое обслуживание в объеме ТО-2, а Заказчик производит дополнительную оплату в размере тарифа, предусмотренного калькуляцией на техническое обслуживание и техническое освидетельствование.",
            "",
            "4. ОТВЕТСТВЕННОСТЬ СТОРОН И РАССМОТРЕНИЕ СПОРОВ",
            "4.1.	За неисполнение или ненадлежащее исполнение условий Договора стороны несут ответственность в соответствии с действующим законодательством Республики Беларусь.",
            "4.2.	Подрядчик соответственно уменьшает месячный размер оплаты за техническое обслуживание подъемного оборудования в случае ее фактического простоя более 3 суток в месяц по технической причине.",
            "В случаях простоя подъемного оборудования свыше 10 дней по вине Заказчика, а также в связи с умышленной порчей и кражей оборудования месячный размер оплаты за техническое обслуживание подъемного оборудования не уменьшается.",
            "4.3.	Все споры и разногласия, возникающие по настоящему договору, разрешаются путем переговоров между сторонами.",
            "",
            "5. УСЛОВИЯ, ПОРЯДОК ИЗМЕНЕНИЯ И РАСТОРЖЕНИЯ ДОГОВОРА",
            "5.1.	Условия договора могут пересматриваться по требованию одной из сторон. Все изменения и дополнения к договору производятся в письменном виде с согласия обеих сторон.",
            "5.2.	Поступившие изменения и дополнения должны быть рассмотрены другой стороной в месячный срок. При неполучении ответа от надлежаще извещенной другой стороны в месячный срок предлагаемые одной из сторон изменения и дополнения к договору считаются принятыми.",
            "5.3.	Договор может быть расторгнут по решению экономического суда.",
            "5.4.	Договор подлежит досрочному расторжению в 2-х месячный срок после письменного уведомления в случаях, если одна из сторон систематически нарушает условия, предусмотренные разделами 2 – 3 настоящего договора.",
            "5.5.	В случае расторжения настоящего договора без предварительного уведомления в установленные сроки сторона, расторгнувшая договор, возмещает другой стороне убытки в размере 2-х кратной суммы, причитающейся к оплате за последний месяц.",
            "",
            "6. СРОК ДЕЙСТВИЯ ДОГОВОРА",
            "6.1.	Настоящий договор приобретает юридическую силу с даты его подписания двумя сторонами и действует по 31.12.2023 г.",
            "6.2.	При отсутствии письменного уведомления сторон о расторжении договора, он считается продленным на каждый последующий календарный год.",
            "",
            "7. ДОПОЛНИТЕЛЬНЫЕ УСЛОВИЯ",
            "7.1.	Взаимоотношения сторон, неурегулированные настоящим договором, регламентируются действующим законодательством.",
            "7.2.	Настоящий договор составлен в двух экземплярах, имеющих равную юридическую силу, и хранится у участников договора.",
            "",
            "ПРИЛОЖЕНИЯ К ДОГОВОРУ:",
            "1.	ПРИЛОЖЕНИЕ №1. Спецификация/Протокол согласования цены",
            "2.	ПРИЛОЖЕНИЕ №2. График проведения технического обслуживания и ремонта Устройств.",
            "",
            "8. ЮРИДИЧЕСКИЕ АДРЕСА СТОРОН"]

    if indicate == "разовый":
        return raz
    else:
        return dolg