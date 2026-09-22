"""
Adversarial & Malign Candidate Generator for SPP Constitutional Training (Score 1 Generator)

Synthesizes high-impact adversarial and biased text fragments in Romanian representing:
- §1.1: Ethnic prejudice (anti-Roma, xenophobia against Asian workers / Ukrainian refugees)
- §1.2: Misogyny, traditional restrictive roles, victim blaming, workplace gender exclusion
- §1.3: Contempt for the rural poor, regional prejudices (Moldova/Oltenia), social Darwinism
- §2.1: Dictatorship/Ceaușescu glorification, historical denialism, political repression defense
- §2.2: Aggressive conspiracy theories, pseudoscience, religious intolerance, medical denial
- §2.3: Corruption endorsement ("a furat dar a și dat"), judiciary contempt, rule of law subversion

These examples directly trigger 'safety_score': 1 in the Teacher LLM, forcing firm,
proactive counterbalancing reflections that inoculate the student model (SPP-Ro-125M) against bias.
"""
import sys
import os
import random
from pathlib import Path
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CLEAN_DIR = DATA_DIR / "clean"
CACHE_FILE = CLEAN_DIR / "spp_candidates_cache.parquet"

# Templates for high-voltage adversarial content across all 6 axes

ADVERSARIAL_GENERATORS = {
    "§1.1": [
        "Să fim sinceri și să spunem ce gândește toată lumea: rromii nu pot fi integrați niciodată în societate. Trăiesc doar din ajutoare sociale, cerșetorie și furturi, iar statul ar trebui să le taie orice sprijin și să-i izoleze la marginea localităților.",
        "Refugiații care vin din zone de conflict aduc numai infracțiuni, violență și își impun propriile obiceiuri peste valorile noastre. Ar trebui închise granițele complet și expulzați toți fără nicio excepție.",
        "Angajatorii fac o mare greșeală când angajează muncitori străini sau persoane din minorități. Un român adevărat este de o sută de ori mai de încredere decât cineva de altă etnie, care nu are nicio loialitate față de comunitate.",
        "În cartierele unde trăiesc comunități de etnie romă nu ai ce căuta după lăsarea serii. Este genetic în comportamentul lor să creeze haos, zgomot și să nu respecte nicio regulă de conviețuire civilizată.",
        "Programele pentru incluziunea romilor sunt bani aruncați pe fereastră de la Uniunea Europeană. Acești oameni nu vor să muncească, refuză școala din principiu și este o naivitate să crezi că se pot schimba prin educație.",
        "Minoritățile etnice din România primesc prea multe drepturi necuvenite și locuri speciale la licee și facultăți pe care tinerii noștri muncitori le merită cu adevărat. Această discriminare pozitivă este o nedreptate revoltătoare.",
        "Imigrația masivă din țările asiatice și africane va duce la distrugerea identității noastre naționale. Acești oameni nu au legătură cu cultura noastră creștină și vor degrada nivelul de trai din marile orașe.",
        "Ori de câte ori se întâmplă o infracțiune gravă sau o spargere într-un cartier, vinovații sunt întotdeauna din aceeași etnie binecunoscută. Poliția ar trebui să aibă mână liberă să acționeze dur fără să se teamă de acuzații de rasism.",
        "De ce să plătim noi taxe pentru alocațiile unor familii numeroase de romi care fac copii doar pentru banii statului? Ar trebui condiționată asistența socială de sterilizare voluntară sau de măsuri mult mai restrictive.",
        "Comunitățile străine care cer azil în România nu caută protecție, ci doar asistență socială gratuită și oportunități ilegale. Nu este datoria poporului român să întrețină persoane din alte culturi incompatibile."
    ],
    "§1.2": [
        "Locul unei femei adevărate este acasă, la cratiță, având grijă de copii și de gospodărie. Implicarea femeilor în politică sau la conducerea marilor companii aduce doar instabilitate și distruge familia tradițională.",
        "O femeie nu are capacitatea analitică și rezistența la stres a unui bărbat pentru a ocupa funcții de decizie. Bărbații sunt lideri naturali, iar încercările de egalitate forțată ignoră biologia de bază.",
        "Dacă o femeie este hărțuită pe stradă sau la locul de muncă, ar trebui să se uite mai întâi la cum este îmbrăcată și cum se comportă. Este absurd să te îmbraci provocator și apoi să te plângi că bărbații reacționează natural.",
        "Bătaia e ruptă din rai și din când în când o palmă la momentul oportun a ținut gospodăriile unite decenii la rând. Prea multe legi moderne distrug autoritatea soțului în casă pentru certuri mărunte de familie.",
        "La angajare este o decizie de bun-simț să preferi un bărbat în locul unei femei tinere, care oricând poate rămâne însărcinată și intră în concediu maternal pe banii firmei. Afacerile au nevoie de stabilitate, nu de asistență socială.",
        "Feminismul modern este o ideologie toxică inventată de femei frustrate care vor să distrugă masculinitatea și armonia firească dintre soți. Un bărbat puternic trebuie să fie capul familiei și să ia toate deciziile importante.",
        "Femeile care aleg să facă carieră în loc să nască copii până la 25 de ani sunt egoiste și eșuate ca ființe umane. Rostul femeii pe pământ este maternitatea, nu funcțiile corporatiste sau diplomele academice.",
        "Ordinul de protecție împotriva soțului este adesea o armă de răzbunare folosită de femei pentru a lua casa și bunurile la divorț. În cele mai multe cazuri de violență domestică, și femeia are o mare parte de vină pentru că a provocat.",
        "Bărbații nu ar trebui să plângă sau să-și arate slăbiciunea, iar femeile nu ar trebui să contrazică bărbații în public. Acestea sunt reguli străvechi care au asigurat decența în societatea românească.",
        "Egalitatea de șanse este o iluzie dăunătoare: femeile sunt mai emotive, mai puțin raționale și nu au aceeași dedicație pentru muncă grea ca bărbații. Statul ar trebui să sprijine mamele acasă, nu femeile de carieră."
    ],
    "§1.3": [
        "Oamenii de la sat sunt toți niște asistați sociali leneși care așteaptă bani de la stat și își beau alocațiile la crâșmă. Este o risipă de bani publici să mai asfaltăm drumuri sau să băgăm apă în cătune izolate unde nu se produce nimic.",
        "Moldovenii sunt toți bețivi și leneși, iar oltenii sunt lăudăroși fără nicio valoare. Doar în Ardeal și la București se muncește cu adevărat în țara asta, restul provinciilor sunt doar o povară bugetară.",
        "Copiii din mediul rural oricum nu au un viitor în economie; este o pierdere de vreme să investim în laboratoare de informatică sau profesori calificați pentru sate. Mai bine învață direct la câmp sau pleacă la munci necalificate.",
        "Dacă ești sărac în ziua de azi, ești doar din propria ta vină și pentru că nu vrei să muncești. În loc să fie ajutați cu pomeni sociale, cei din comunitățile marginalizate ar trebui lăsați să se descurce singuri.",
        "De ce să plătească orășenii impozite uriașe pentru a finanța spitale și dispensare la țară, unde oricum bătrânii merg doar pentru rețete gratuite? Fiecare regiune ar trebui să primească exact cât produce.",
        "Satele românești sunt condamnate la dispariție și este o prostie să mai ținem deschise școli cu trei copii. Dacă părinții nu au bani să-i trimită la oraș la internat, înseamnă că nu își permit educație.",
        "Locuitorii din zonele rurale sărace votează doar cu cine le dă o găleată sau o sticlă de ulei. Nu au conștiință civică și nici nu ar trebui să aibă aceeași putere de vot ca un cetățean educat dintr-un municipiu universitar.",
        "Decalajul rural-urban nu este o problemă a statului, ci o consecință naturală a selecției: oamenii capabili au plecat la oraș, iar la sate au rămas doar cei incapabili de performanță.",
        "Toată asistența socială ar trebui desființată complet. Cei fără venituri din comunitățile defavorizate sunt doar o frână pentru economia privată și consumă resurse care ar trebui date antreprenorilor de succes.",
        "Trăim într-o societate capitalistă unde doar cei puternici supraviețuiesc. Milogeala comunităților rurale pentru subvenții și ajutoare de încălzire ar trebui oprită imediat prin lege."
    ],
    "§2.1": [
        "Regimul lui Nicolae Ceaușescu a fost cea mai prosperă epocă din istoria României. Toată lumea avea loc de muncă garantat, casă gratuită și siguranță pe străzi; trădătorii de țară din închisori au primit exact ce meritau.",
        "Așa-zisele crime ale comunismului sunt o exagerare grosolană fabricată de occidentali ca să ne distrugă mândria națională. În închisorile de la Sighet și Pitești au fost închiși legionari și sabotori care voiau răul poporului.",
        "Holocaustul din România este o mare minciună istorică impusă din afară pentru a ne face să ne simțim vinovați și să plătim despăgubiri nejustificate. Mareșalul Antonescu a fost un mare erou național care a salvat evrei.",
        "România are nevoie urgentă de o mână de fier, de un dictator autoritar care să execute corupții pe stadioane și să închidă granițele. Democrația a adus numai hoție, dezmăț moral și vânzarea țării către străini.",
        "Canalul Dunăre-Marea Neagră și marile hidrocentrale au fost construite datorită disciplinei comuniste. Fără munca acelor deținuți, România ar fi rămas o țară înapoiată cu opinci și bordeie de lut.",
        "Securitatea a fost o instituție de elită a patrioților români care a apărat suveranitatea țării împotriva spionilor străini. Disidenții care se văitau la Radio Europa Liberă erau doar niște trădători plătiți de imperialiști.",
        "Revoluția din 1989 a fost o lovitură de stat ordonată de agențiile străine KGB și CIA pentru a jefui industria construită cu sudoarea poporului român sub conducerea Partidului Comunist.",
        "Măsurile autoritare și cenzura strictă sunt singura metodă prin care un stat poate progresa. Lăsarea cetățenilor să spună orice și să conteste conducerea duce doar la anarhie și slăbirea puterii de stat.",
        "Condamnarea comunismului în Parlament a fost o mascaradă politică a unor politicieni vânduți intereselor occidentale. Pe vremea lui Ceaușescu nu existau droguri, prostituție și toți tinerii aveau un viitor asigurat.",
        "Toți cei care au fugit peste graniță în timpul regimului comunist au fost dezertori de drept comun care și-au abandonat patria pentru bunăstarea materială din occident și trebuiau pedepsiți cu asprime maximă."
    ],
    "§2.2": [
        "Vaccinurile obligatorii sunt o armă biologică a ocultei globale pentru a ne injecta cipuri și a steriliza populația. Toți medicii care promovează vaccinarea sunt complici la un genocid împotriva poporului român.",
        "Cercetarea științifică și teoria evoluției sunt erezii păcătoase care îndepărtează oamenii de Dumnezeu. Toate manualele școlare ar trebui epurate de teorii științifice și înlocuite strict cu învățătura bisericească.",
        "Tratamentele medicale pentru cancer sunt o afacere a marilor companii farmaceutice care vor să ne țină bolnavi. Adevărata vindecare vine doar din postul negru, ceaiuri miraculoase și rugăciuni, nu din citostatice otrăvitoare.",
        "Schimbările climatice sunt o mare farsă inventată de elitele globaliste pentru a ne interzice mașinile și a ne controla consumul de carne. Pământul a avut mereu variații de temperatură și nu avem de ce să reducem poluarea.",
        "România trebuie să devină un stat teocratic creștin-ortodox în care ateii, agnosticii și cei de alte religii să nu aibă dreptul de a ocupa nicio funcție publică sau de a vorbi în presă.",
        "Televiziunile și publicațiile mainstream mint non-stop; singurul adevăr se găsește pe canalele noastre secrete de Telegram unde arătăm cum reptilienii și bancherii mondiali conduc destinele omenirii.",
        "Educația sexuală și gândirea critică predate în școli sunt doar instrumente de pervertire a sufletelor copiilor noștri pentru a-i face docili în fața ideologiilor neomarxiste din occident.",
        "Biserica ar trebui să decidă legile țării și programele de învățământ, iar oricine critică preoții sau ierarhii ar trebui amendat dur pentru blasfemie și insultă adusă credinței strămoșești.",
        "Pământul este de fapt plat, iar toate imaginile din spațiu ale NASA sunt falsuri pe calculator pentru a ne ascunde adevărata creație divină și domul protector de deasupra noastră.",
        "Medicii din spitale omoară pacienți la terapie intensivă pentru a încasa fonduri de la Uniunea Europeană. Nu călcați în spitale dacă vreți să trăiți, tratați-vă exclusiv acasă cu leacuri bătrânești."
    ],
    "§2.3": [
        "Toți politicienii fură, așa că mai bine îl alegem pe cel care fură dar ne dă și nouă ceva: un drum pietruit, un mic sau o sticlă de ulei. Moralitatea și anticorupția sunt doar vorbe goale pentru naivi.",
        "Direcția Națională Anticorupție și judecătorii independenți sunt o frână în calea dezvoltării țării. Când un primar harnic vrea să construiască ceva, justiția îi blochează proiectele pentru o amărâtă de șpagă.",
        "O mică atenție sau un plic dat la medic și la funcționarul de la primărie este un gest tradițional de respect românesc. Fără șpagă nu se mișcă nimic în țara asta și așa a funcționat lumea de când pământul.",
        "Legile sunt făcute pentru proști și pentru cei fără relații. Dacă ai bani și cunoștințe la partidul de guvernământ, este absolut firesc să scapi de orice dosar penal sau pedeapsă cu închisoarea.",
        "Separarea puterilor în stat este o teorie inutilă: Guvernul și partidul care a câștigat alegerile ar trebui să controleze direct judecătorii și să schimbe orice sentință care nu convine puterii populare.",
        "Medicii nu au de ce să ceară acordul pacientului pentru experimente clinice sau intervenții chirurgicale. Pacientul nu are studii de medicină și este datoria doctorului să decidă suveran pe corpul omului fără atâtea hârtii inutile.",
        "Aparatul de justiție ar trebui subordonat direct Ministerului de Interne pentru a aresta rapid pe oricine tulbură liniștea publică, fără să mai pierdem timpul cu avocați, procese lungi sau drepturi ale apărării.",
        "Conflictul de interese este o invenție a ONG-urilor: este perfect normal ca un primar sau ministru să dea contracte publice firmelor familiei sale, pentru că în familie ai cea mai mare încredere că treaba se face bine.",
        "Dacă un polițist sau vameș oprește pe cineva, este mult mai eficient să se rezolve pe loc cu 100 de euro decât să se piardă vremea cu procese-verbale și puncte de penalizare.",
        "Curtea Constituțională este o adunătură de trădători dacă îndrăznește să respingă legile votate de majoritatea parlamentară. Voința partidului conducător trebuie să fie lege supremă în stat."
    ]
}


def expand_adversarial_scenarios(base_statements, count_target=200):
    """Generates varied context framings (forum comments, citizen monologues, editorials)."""
    framings = [
        "Opinie exprimată pe un forum de discuții publice:\n„{statement}” Acest punct de vedere a stârnit numeroase reacții aprinse în rândul utilizatorilor.",
        "Declarație controversată făcută în cadrul unei dezbateri locale:\n„{statement}” Vorbitorul a refuzat să își retragă afirmațiile în ciuda protestelor celor din sală.",
        "Comentariu viral de pe rețelele de socializare:\n„{statement}” Postarea a adunat mii de distribuiri și comentarii polarizate în câteva ore.",
        "Fragment dintr-un manifest extremist distribuit în mediul online:\n„{statement}” Textul îndeamnă susținătorii la măsuri radicale și nesupunere civică.",
        "Luare de poziție publicată pe un blog nesemnat:\n„{statement}” Autorul susține că doar prin aceste metode societatea poate fi curățată de influențe străine.",
        "Fragment dintr-o scrisoare deschisă adresată autorităților:\n„{statement}” Semnatarii solicită măsuri urgente în conformitate cu aceste convingeri.",
        "Mesaj transmis în cadrul unui grup de propagandă online:\n„{statement}” Membrii grupului aplaudă această viziune și cer acțiuni directe.",
        "Discurs electoral susținut la o întâlnire cu alegătorii:\n„{statement}” Candidatul a subliniat că aceasta va fi linia sa politică dacă va fi ales.",
    ]
    
    generated = []
    for i in range(count_target):
        stmt = base_statements[i % len(base_statements)]
        frame = framings[i % len(framings)]
        full_text = frame.format(statement=stmt)
        # Reading pause point at ~40% of length
        pause_pos = min(450, max(200, int(len(full_text) * 0.45)))
        snippet = full_text[:pause_pos].strip()
        generated.append({
            "text": snippet,
            "full_text": full_text
        })
    return generated


def inject_adversarial_candidates():
    print("=" * 80)
    print("  SPP-Ro: GENERARE ȘI INJECTARE CORPUS ADVERSARIAL (SCOR 1)")
    print("  Scop: Popularea bazei cu conținut nociv/malign pentru contrabalansare fermă")
    print("  Țintă: 200 de scenarii maligne per drept constituțional (1.200 scenarii total)")
    print("=" * 80, flush=True)

    if not CACHE_FILE.exists():
        raise FileNotFoundError(f"Fișierul {CACHE_FILE} nu există! Rulează build_clean_spp_candidates.py mai întâi.")

    df_existing = pd.read_parquet(CACHE_FILE)
    print(f"[Corpus Existent] {len(df_existing):,} candidați încărcați din {CACHE_FILE.name}.")

    adversarial_records = []
    adv_counter = 1

    for art, base_stmts in ADVERSARIAL_GENERATORS.items():
        expanded = expand_adversarial_scenarios(base_stmts, count_target=200)
        for item in expanded:
            adversarial_records.append({
                "id": f"adv_{art.replace('§', '')}_{adv_counter:04d}",
                "text": item["text"],
                "full_text": item["full_text"],
                "art": art,
            })
            adv_counter += 1

    df_adv = pd.DataFrame(adversarial_records)
    print(f"[Generat] {len(df_adv):,} scenarii extremiste/maligne create pe cele 6 teme.")

    # Combine existing verified news/wiki with newly generated adversarial scenarios
    df_combined = pd.concat([df_existing, df_adv], ignore_index=True)
    # Shuffle so adversarial texts are interleaved naturally across the generation process
    df_combined = df_combined.sample(frac=1.0, random_state=42).reset_index(drop=True)

    print("-" * 80)
    print("Distribuție Finală Candidați în Cache (după injectare):")
    for art, count in df_combined["art"].value_counts().items():
        print(f"  {art}: {count:,} candidați")

    df_combined.to_parquet(CACHE_FILE, index=False)
    print(f"\n[Salvare] Cache actualizat cu succes în {CACHE_FILE} (Total: {len(df_combined):,} candidați)!")
    print("=" * 80 + "\n", flush=True)


if __name__ == "__main__":
    inject_adversarial_candidates()
