nbutton_widget_instruction = '''
---

**Istruzioni per l’uso dello strumento dei pulsanti**

Per utilizzare questo strumento, includi nel tuo messaggio un blocco formattato in questo modo (NON GENERARE ANCHE I CARATTERI "```" MA SOLO IL CONTENUTO INTERNO!):

```
< TYPE='WIDGET' WIDGET_ID='NButtonWidget' | {{"buttons": [
  ["label": "Opzione 1", "reply": "Risposta per opzione 1"}},
  {{"label": "Opzione 2", "reply": "Risposta per opzione 2"}},
  {{"label": "Opzione 3", "reply": "Risposta per opzione 3"}},
  {{"label": "Opzione 4", "reply": "Risposta per opzione 4"}},
  {{"label": "Opzione 5", "reply": "Risposta per opzione 5"}},
  {{"label": "Opzione 6", "reply": "Risposta per opzione 6"}}
]}} | TYPE='WIDGET' WIDGET_ID='NButtonWidget' >
```

Assicurati che:

- Il blocco inizi con `< TYPE='WIDGET'` e termini con `>`.
- All’interno del blocco, il JSON (compreso tra due caratteri `|`) contenga un array di oggetti, ognuno con le proprietà `label` (l’etichetta del pulsante) e `reply` (la risposta da inviare quando il pulsante viene premuto).

Il sistema rileverà questo pattern, sostituirà il blocco con un placeholder e genererà un widget interattivo in cui i pulsanti saranno distribuiti uniformemente in una griglia.

Quando l’utente clicca su uno di questi pulsanti, la risposta associata (specificata nel campo `reply`) verrà inviata al chatbot per ulteriori elaborazioni.

---
'''

change_chat_name_instructions = """
---
**Istruzioni per lo strumento "ChangeChatNameWidget"**

Per utilizzare questo strumento, includi nel tuo messaggio un blocco formattato esattamente come segue (non includere i caratteri backtick; usa solo il contenuto interno):

< TYPE='WIDGET' WIDGET_ID='ChangeChatNameWidget' | {{
  "chatId": "ID_DELLA_CHAT",
  "newName": "NUOVO_NOME_CHAT"
}} | TYPE='WIDGET' WIDGET_ID='ChangeChatNameWidget' >

**Dettagli:**
- **"chatId"**: l’identificativo univoco della chat da rinominare. Se non viene fornito alcun valore (cioè se il campo è una stringa vuota `""`), il sistema utilizzerà automaticamente l’ID della chat attuale.
- **"newName"**: il nuovo nome da assegnare alla chat.

Quando il chatbot invia questo blocco, il sistema lo riconoscerà ed eseguirà immediatamente l’operazione di rinominazione senza richiedere ulteriori conferme. Dopo l’esecuzione, verrà visualizzata in chat una scheda di conferma con uno sfondo verde sfumato contenente:
- Il titolo “OPERAZIONE EFFETTUATA”.
- Un messaggio che indica che la chat è stata rinominata. Se il campo "chatId" era vuoto, il messaggio farà riferimento alla chat attuale (mostrando l’ID ottenuto automaticamente); altrimenti, verrà riportato l’ID specificato nel blocco.

Assicurati che il tuo codice implementi la logica di rinominazione in modo che il nuovo nome venga aggiornato in maniera persistente, sia localmente (stato e localStorage) sia nel database remoto.

---
"""

chat_vars_widget_instructions = """
---
🎛 **Istruzioni per lo strumento "ChatVarsWidget"**

Usalo per creare / aggiornare variabili di stato **persistenti** della chat
(denominate *chatVars*).  
Includi **esattamente** questo blocco (senza back-tick):

< TYPE='WIDGET' WIDGET_ID='ChatVarsWidget' | {{
  "updates": {{
    "chiave1": "valore qualunque",          // stringa
    "counter": 42,                          // numero
    "flagAttivo": true,                     // boolean
    "oggettoComplesso": {{                 // sotto-oggetto JSON
      "campoA": "abc",
      "campoB": [1, 2, 3]
    }}
  }}
}} | TYPE='WIDGET' WIDGET_ID='ChatVarsWidget' >

📌 **Dettagli**  
* `WIDGET_ID` deve essere `ChatVarsWidget`.  
* **`updates`** è un oggetto arbitrario; ogni coppia *chiave → valore* va  
  ad aggiornare (o creare) la stessa chiave dentro *chatVars*.  
  Le chiavi già esistenti vengono sovrascritte; le altre rimangono intatte.  
* Dopo l’esecuzione il chatbot mostra una piccola card di conferma che  
  scompare da sola.

**Esempio rapido**

< TYPE='WIDGET' WIDGET_ID='ChatVarsWidget' | {{
  "updates": {{
    "todoDone": false,
    "lastSummaryDate": "2025-06-07"
  }}
}} | TYPE='WIDGET' WIDGET_ID='ChatVarsWidget' >
---
"""

auto_sequence_widget_instructions = """
---

**Istruzioni complete per lo strumento “AutoSequenceWidget”**

### 🔍  Perché esiste  
L’agente può compiere in **autonomia** operazioni che richiedono **più turni consecutivi** senza dover chiedere di volta in volta nuove istruzioni all’utente.  
Con **AutoSequenceWidget** il chatbot genera in anticipo la **lista di messaggi** che *dovrebbe* ricevere dall’utente per completare un’attività complessa; la UI li invierà in sequenza, come se l’utente li digitasse uno dopo l’altro, subito dopo che l’assistente ha risposto al turno precedente.  
In questo modo si superano i limiti di un singolo prompt e si evitano richieste manuali ripetitive.

### ✍️  Sintassi del blocco  
*(NON inserire caratteri back-tick ``` nel blocco, né la chiave `is_first_time` – il sistema la gestisce da sé)*  

```
< TYPE='WIDGET' WIDGET_ID='AutoSequenceWidget' | {{
  "sequence": [
    {{ "message": "testo che l’assistente DEVE ricevere al passo 1" }},
    {{ "message": "testo del passo 2" }},
    …
    {{ "message": "testo del passo n" }}
  ]
}} | TYPE='WIDGET' WIDGET_ID='AutoSequenceWidget' >
```

* Campi obbligatori  
  * **`WIDGET_ID`**deve essere **`AutoSequenceWidget`**  
  * **`sequence`**array ordinato di oggetti  
      * `message`(stringa) contenuto da inviare all’assistente a quel passo  
      * `delay_ms`(numero, *ignorato*) – può esserci ma non ha effetto  

* Regole sintattiche  
  * Il blocco inizia con `< TYPE='WIDGET'` e termina con `>`  
  * Non usare virgolette attorno al blocco, né racchiuderlo in altri tag  

### ⚙️  Come funziona internamente  
1. **Creazione** – L’assistente include il blocco nel proprio messaggio.  
2. **Salvataggio** – Il sistema salva la conversazione; il widget appare.  
3. **Avvio automatico** – Dopo che l’assistente ha **completato** quel turno, il widget:  
   * invia il **primo** `message` come se fosse l’utente;  
   * attende che l’assistente risponda;  
   * prosegue con il messaggio successivo, fino a esaurimento della sequenza.  
4. **Una sola esecuzione** – Il widget forza `is_first_time:false` così, se la chat viene ricaricata, la sequenza **non** riparte.

### 🤖  Quando utilizzarlo  
| Caso d’uso | Perché è utile |
|------------|---------------|
| Ricerche o analisi strutturate suddivise in più fasi | Permette di guidare l’agente attraverso ogni fase senza ulteriori input umani |
| Workflow passo-passo (brainstorm → outline → bozza → revisione) | Automatizza la catena di prompt e garantisce che l’assistente segua tutte le tappe |
| Attività iterative su documenti / codice | Fa sì che l’assistente riceva le istruzioni in ordine senza dimenticare nulla |

### 🖼️  Esempio pratico  
**Scenario**: l’utente chiede *“Confronta in profondità gli articoli 4-8 della Direttiva A con gli articoli 5-9 della Direttiva B nella mia Knowledge Box, evidenzia le differenze sottili e scrivi una relazione finale.”*  
L’agente può rispondere così:

< TYPE='WIDGET' WIDGET_ID='AutoSequenceWidget' | {{
  "sequence": [
    {{ "message": "Step 1 – Recupera da KB i testi integrali degli articoli 4-8 della Direttiva A e 5-9 della Direttiva B." }},
    {{ "message": "Step 2 – Elenca in tabella gli articoli affiancati con differenze di wording evidenziate." }},
    {{ "message": "Step 3 – Evidenzia le differenze **concettuali** (non solo lessicali) e gli effetti pratici." }},
    {{ "message": "Step 4 – Scrivi una relazione conclusiva di max 400 parole con riferimenti alla KB." }}
  ]
}} | TYPE='WIDGET' WIDGET_ID='AutoSequenceWidget' >

Una volta inviato:  
1. Il widget spedirà il messaggio *Step 1* → l’assistente cercherà i testi nella KB e risponderà.  
2. Alla chiusura della risposta, il widget invierà *Step 2*, e così via fino al *Step 4*.  
3. Alla fine il widget mostrerà “Sequenza completata ✅” e l’utente riceverà l’output finale senza dover intervenire manualmente.

### ✅  Best practice  
* Mantieni i messaggi **auto-contenuti**: ogni passo deve offrire all’assistente le informazioni necessarie per completarlo.  
* Usa **massimo 6-8 step** per non allungare eccessivamente la conversazione.  
* Se servono ricerche aggiuntive via vector store, pianificale nei primi step.  
* Evita sovrapposizioni: ogni step deve avere un obiettivo diverso.  
* Non inserire logiche condizionali complesse: la sequenza è lineare.

---
"""

js_runner_instructions = """

---

**Istruzioni per lo strumento "JSRunnerWidget"**

Per inserire un blocco che permetta all’utente di eseguire codice JavaScript:

< TYPE='WIDGET' WIDGET_ID='JSRunnerWidget' | {{
  "code"  : "// Scrivi qui il tuo JavaScript (console.log('Hi');)",
  "height": 350   // Altezza in pixel (facoltativa, default = 300)
}} | TYPE='WIDGET' WIDGET_ID='JSRunnerWidget' >

**Linee guida**
1.  Il parametro **`WIDGET_ID`** deve essere esattamente `"JSRunnerWidget"`.
2.  Campi JSON supportati  
    - **"code"** (stringa) Il codice JS da eseguire nell’iframe.  
    - **"height"** (numero) Altezza area iframe; se omesso vale 300 px.
3.  Il widget apre un iframe isolato con sandbox `allow-scripts`; può contenere `console.log`, fetch, librerie esterne via `<script src="…">`, ecc.
4.  Scorrimento fluido  
    - L’iframe propaga gli eventi di _scroll_ al contenitore di chat: quando l’utente raggiunge l’inizio o la fine dell’iframe, lo scroll continua nella chat senza “bloccarsi”.
5.  **Non** aggiungere la chiave `"is_first_time"`: la gestisce il sistema.

Esempio rapidissimo:

< TYPE='WIDGET' WIDGET_ID='JSRunnerWidget' | {{
  "code": "console.log('Hello World!'); alert('Eseguito!')",
  "height": 400
}} | TYPE='WIDGET' WIDGET_ID='JSRunnerWidget' >

========================================================
IMPORTANTISSIMO!!!!: IL CODICE JS DA TE FORNITO VERRà ESEGUITO IN SEGUENTE ELEMENTO HTML
'''
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
  <!-- 1. Carica ECharts da CDN -->
  <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
  html,body{{margin:0;height:100%;overflow:auto;font-family:monospace;}}
</style>
</head>
<body>
<pre id="out"></pre>
<!-- Aggiungi questo div per il grafico -->
<div id="chart" style="width: 600px; height: 400px;"></div>
<script>

try{{ 
  const log=(...a)=>document.getElementById('out').textContent+=a.join(' ')+'\\n';
  console.log=log;console.error=log;
  ${{......YOUR INPUT CODE........}}
}}catch(e){{console.error(e);}}
</script>
</body>
</html>
''';
DUNQUE TU DOVRAI FORNIRE UN SINGOLO SCRIPT MA SENZA USARE ASSOLUTAMENTE CARATTERI '<' E '>' ALL'IINTERNO DELL INPUT FONRITO AL WIDGET, ALTRIMENTI ANDRESTI IN CONFLITTO CON SISTEMA DI APRSING CHE RICONOSCE IL WIDGET. ASSICURATI DI RISPETTARE QUESTA REGOLE.

INOLTRE DOVRAI SEMPRE USARE ECHARTS PER GENERARE GRAFICI E ELEMENTI DI RAPPRESENTAIOZNE DEI DATI.

DIS EGUITO ALCUNI ESMEPI DI UTILIZZO:
< TYPE='WIDGET' WIDGET_ID='JSRunnerWidget' | {{"code":"document.addEventListener('DOMContentLoaded', function() {{
  const chartDom = document.getElementById('chart');
  if (chartDom) {{
    const myChart = echarts.init(chartDom);

    const option = {{
      title: {{
        text: 'Vendite Mensili',
        left: 'center'
      }},
      tooltip: {{
        trigger: 'axis'
      }},
      xAxis: {{
        type: 'category',
        data: ['Gen', 'Feb', 'Mar', 'Apr', 'Mag', 'Giu', 'Lug', 'Ago', 'Set', 'Ott', 'Nov', 'Dic']
      }},
      yAxis: {{
        type: 'value'
      }},
      series: [{{
        name: 'Vendite',
        type: 'bar',
        data: [150, 200, 180, 220, 170, 250, 300, 280, 260, 310, 290, 330],
        markPoint: {{
          data: [
            {{ type: 'max', name: 'Massimo' }},
            {{ type: 'min', name: 'Minimo' }}
          ]
        }},
        markLine: {{
          data: [
            {{ type: 'average', name: 'Media' }}
          ]
        }}
      }}]
    }};

    myChart.setOption(option);
  }} else {{
    console.error('Elemento con id \\"chart\\" non trovato.');
  }}
}});","height":400}} | TYPE='WIDGET' WIDGET_ID='JSRunnerWidget' >
---------

< TYPE='WIDGET' WIDGET_ID='JSRunnerWidget' | {{"code":"document.addEventListener('DOMContentLoaded', function() {{
  const chartDom = document.getElementById('chart');
  if (chartDom) {{
    const myChart = echarts.init(chartDom);

    const option = {{
      title: {{
        text: 'Vendite Mensili',
        left: 'center'
      }},
      tooltip: {{
        trigger: 'axis'
      }},
      xAxis: {{
        type: 'category',
        data: ['Gen', 'Feb', 'Mar', 'Apr', 'Mag', 'Giu', 'Lug', 'Ago', 'Set', 'Ott', 'Nov', 'Dic']
      }},
      yAxis: {{
        type: 'value'
      }},
      series: [{{
        name: 'Vendite',
        type: 'bar',
        data: [150, 200, 180, 220, 170, 250, 300, 280, 260, 310, 290, 330],
        markPoint: {{
          data: [
            {{ type: 'max', name: 'Massimo' }},
            {{ type: 'min', name: 'Minimo' }}
          ]
        }},
        markLine: {{
          data: [
            {{ type: 'average', name: 'Media' }}
          ]
        }}
      }}]
    }};

    myChart.setOption(option);
  }} else {{
    console.error('Elemento con id \\"chart\\" non trovato.');
  }}
}});","height":400}} | TYPE='WIDGET' WIDGET_ID='JSRunnerWidget' >
--------

< TYPE='WIDGET' WIDGET_ID='JSRunnerWidget' | {{"code":"document.addEventListener('DOMContentLoaded', function() {{
  const chartDom = document.getElementById('chart');
  if (chartDom) {{
    const myChart = echarts.init(chartDom);

    const option = {{
      title: {{
        text: 'Diagramma di Flusso a Nodi Avanzato',
        left: 'center'
      }},
      tooltip: {{}},
      series: [{{
        type: 'graph',
        layout: 'force',
        symbolSize: 70,
        roam: true,
        label: {{
          show: true
        }},
        force: {{
          repulsion: 300,
          gravity: 0.1,
          edgeLength: [100, 200]
        }},
        edgeSymbol: ['circle', 'arrow'],
        edgeSymbolSize: [4, 10],
        edgeLabel: {{
          fontSize: 12
        }},
        data: [
          {{ name: 'Nodo A' }},
          {{ name: 'Nodo B' }},
          {{ name: 'Nodo C' }},
          {{ name: 'Nodo D' }},
          {{ name: 'Nodo E' }},
          {{ name: 'Nodo F' }}
        ],
        links: [
          {{ source: 'Nodo A', target: 'Nodo B' }},
          {{ source: 'Nodo A', target: 'Nodo C' }},
          {{ source: 'Nodo B', target: 'Nodo D' }},
          {{ source: 'Nodo C', target: 'Nodo D' }},
          {{ source: 'Nodo D', target: 'Nodo E' }},
          {{ source: 'Nodo E', target: 'Nodo F' }},
          {{ source: 'Nodo F', target: 'Nodo A' }}
        ],
        lineStyle: {{
          opacity: 0.9,
          width: 2,
          curveness: 0.3
        }}
      }}]
    }};

    myChart.setOption(option);
  }} else {{
    console.error('Elemento con id \\"chart\\" non trovato.');
  }}
}});","height":400}} | TYPE='WIDGET' WIDGET_ID='JSRunnerWidget' >
----

< TYPE='WIDGET' WIDGET_ID='JSRunnerWidget' | {{"code":"document.addEventListener('DOMContentLoaded', function() {{
  const chartDom = document.getElementById('chart');
  if (chartDom) {{
    const myChart = echarts.init(chartDom);

    const option = {{
      title: {{
        text: 'Distribuzione delle Categorie',
        left: 'center'
      }},
      tooltip: {{
        trigger: 'item',
        formatter: '{{b}}: {{c}} ({{d}}%)'
      }},
      legend: {{
        top: 'bottom'
      }},
      series: [{{
        name: 'Categorie',
        type: 'pie',
        radius: '55%',
        data: [
          {{ value: 40, name: 'Categoria A' }},
          {{ value: 30, name: 'Categoria B' }},
          {{ value: 20, name: 'Categoria C' }},
          {{ value: 10, name: 'Categoria D' }}
        ],
        emphasis: {{
          itemStyle: {{
            shadowBlur: 10,
            shadowOffsetX: 0,
            shadowColor: 'rgba(0, 0, 0, 0.5)'
          }}
        }}
      }}]
    }};

    myChart.setOption(option);
  }} else {{
    console.error('Elemento con id \\"chart\\" non trovato.');
  }}
}});","height":400}} | TYPE='WIDGET' WIDGET_ID='JSRunnerWidget' >

===================================================
---
"""

show_chat_vars_widget_instructions = """
---
🔍 **Istruzioni per lo strumento "ShowChatVarsWidget"**

Usalo quando vuoi **mostrare** in chat il valore di una o più variabili
salvate in *chatVars*.

Scrivi **esattamente** il seguente blocco (senza back-tick):

< TYPE='WIDGET' WIDGET_ID='ShowChatVarsWidget' | {{
  "keys": ["var1", "var2"]   // opzionale. Vuoto o mancante = tutte le chiavi
}} | TYPE='WIDGET' WIDGET_ID='ShowChatVarsWidget' >

📌 **Dettagli**
* `WIDGET_ID` **obbligatorio** e deve essere `ShowChatVarsWidget`.
* `keys` è un array di stringhe con i nomi delle chiavi da visualizzare.
  ⁃ Se l’array è vuoto **o** la proprietà non è presente, il widget mostrerà
    **tutte** le chiavi attualmente contenute in *chatVars*.
* Ogni variabile viene renderizzata in un riquadro con:
  ⁃ bordo colorato deterministico (hash del valore),  
  ⁃ sfondo bianco,  
  ⁃ separatore colorato sotto il titolo,  
  ⁃ valore formattato in JSON se strutturato, oppure come testo grezzo.

**Esempio – mostra una singola chiave**

< TYPE='WIDGET' WIDGET_ID='ShowChatVarsWidget' | {{
  "keys": ["todoDone"]
}} | TYPE='WIDGET' WIDGET_ID='ShowChatVarsWidget' >
---
"""

default_instructions = [
    chat_vars_widget_instructions,
    show_chat_vars_widget_instructions,
    nbutton_widget_instruction,
    change_chat_name_instructions,
    auto_sequence_widget_instructions,
    js_runner_instructions,
]