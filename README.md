
# Alerta de EPI e veículos

![Demo of the project](./reademe_imgs/output_video.gif)


Este projeto utiliza a YOLOv8 treinada em um dataset contendo as classes *capacete*,*pessoa* e *veículo* para a tarefa de identificação de pessoas sem EPI e veículos que circulam em uma área restrito. Dessa forma no caso de identificação são realizadas as seguintes ações:

- Emissão de um alerta informando o *ID* da pessoa identificada e o *timestamp* em que occorreu a identificação
- Captura da imagem da pessoa/veículo detectado

## Tabela de Conteúdos
1. Estrutura do Repositório
2. Como Executar
3. Funcionamento do Código

# Extrutura do Repositório:
```bash
Yolov8
├── alertas                            # Pasta contendo os alertas
│   ├── pessoasSemCapacete             # Pasta para salvar os alertas de pessoas sem capacete
│   │   └── alertas.log                # Arquivo de alertas.log para pessoas sem capacete
│   └── veiculos                       # Pasta contendo os alertas para os veículos
│       └── alertas.log                # Arquivo alerta.log para os veículos
├── ch5-cut.mp4                        # vídeo onde é feita a inferência
├── imgs                               # Pasta onde são salvas as imagens
│   ├── PessoasSemCapacete             # Pasta para imagens das pessoas sem capacete
│   │   ├── pessoa_1.png               
│   │   ├── pessoa_2.png               
│   │   └── pessoa_3.png               
│   └── Veiculos                       # Pasta para as imagens dos veículos
├── model                              # Pasta onde são salvos os arquivos contendo os pesos da YOLOv8
│   └── best.pt                        # Pesos da YOLOv8
├── requirements.txt                   # bibliotecas utilizadas
│
│
├── .gitignore                         # Arquivos não necessários
│  
├── README.md                          # Documentação
└── src                                # Arquivos .py
    ├── BaseObject.py                  # Classe base para as classes vehicle e worker
    ├── main.py                        # arquivo principal
    ├── vehicle.py                     # Classe utilizada na detecção dos veículos
    ├── videoAnalyzer.py               # Classe utilizada para para salvar as imagens e gerar alertas
    └── worker.py                      # Classe utiilizada na detecção das pessoas/funcionários

```

# Como Executar

1. **Instalar as dependências:**
   Inicialmente certifique-se de que as dependências necessárias foram instaladas, isso pode ser feito executando o seguinte comando:

   ```bash
   pip install -r requirements.txt
   ```
2. **Navegue até a pasta do projeto:**
     A partir da pasta do projeto execute o comando:
     ```bash
     python main.py
     ```
3. **Saída:**
   - O vídeo será exibido
   - Nele serão mostradas as detecções dos objetos das classes *capacete*,*pessoa* e *veículo*
   - As detecções iram gerar os alertas nos seus devidos arquivos alertas.log além de salvar as imagens


# Funcionamento do Código:
Nesta seção iremos abordar sobre o funcionamento do código. Inicialmente trataremos das classes auxiliares coomo `BaseObject`,`worker` e `vehicle`, então iremos tratar da classe principal,'videoAnalyzer ' utilizada para a detecção e geração dos alertas, por fim mostraremos o fluxo do arquivo principal `main.py`.

## Classes Auxiliares

**1. BaseObject**: Implementa a estrutura principal dos objetos detectados à serem herdados pelas classes `worker` e `vehicle`. 
Nela são criados os atributos:
- *id* ➡️ Salva id do objeto
- *bbox_history* ➡️ armazena coordenadas da bounding box do objeto
- *last_frame_seen* ➡️ número do último frame em que o objeto foi visto
- *frame* ➡️ imagem do objeto detectado
- *frame_timestamp* ➡️ timestamp da detecção

Um único método é implementado por essa classe, sendo ele `add_detection()`, o qual recebe como parâmetros *bbox*, que contém as tuplas da bounding box, e *frame_id* contendo o número do frame em que o objeto foi detectado. Esses parâmetros são adicionados à lista *bbox_history* e atualizada como *last_frame_seen* respectivamente.

**2. vehicle**: A classe `vehicle` apenas herda de `BaseObject`, nela não são feitas nenhuma alterações quanta aos atributos utilizados e ao método herdado.

**3. worker**: A classe `worker`também herda de `BaseObject`, no entanto adiciona o atributo *helmet_status_history* que é uma lista booleana indicando se a pessoa estava ou não utilizando o EPI durante o seu histórico de detecção, consequentemente o método `add_detection()` é acrescido da váriavel *helmet_status* que é um booleano indicando o uso do capacete e este é adicionado à lista mencionada previamente.

## Classe videoAnalyzer
Essa classe tem como atributos:

- *object_type* ➡️ string contendo o tipo do objeto a ser detectado: *pessoa* ou *veículo*
- people ➡️ dicionário utilizado para armazenar as pessoas
- automobile ➡️ dicionário utilizado para armazenar os automóveis
- vehicle_id ➡️ id contendo dos veículos detectados, inicializado como 1
- person_id ➡️ id contendo das pessoas, inicializado como 1

Os métodos utilizados são os seguintes:

- `_get_roi(frame,coordenadas)` ➡️ utilizado para extrair a imagem do objeto detectado de um frame. Os parâmetros *frame* e *coordenadas* se referem ao frame utilizado para extrair a imagem e as coordenadas do bounding box respectivamente.
- `_log_alerts(alert_type, obj_id, timestamp, alert_path)` ➡️ Gera alertas nos arquivos de alertas.log, *alert_type* é uma string *veículo* ou *pessoa*,*obj_id* é o id do objeto, *timestamp* é a timestamp da detecção e *alert_path* é o endereço do arquivo onde será emitido o alerta.
- `_save_imgs(dir_path, img_name, img)` ➡️ Salva as imagens dos objetos detectados, *dir_path* é o diretório onde a imagem será salva, *img_name* é o nome utilizado para salvar a imagem e *img* é a imagem a ser salva.
- `_object_analysis(self, frame, results, current_frame, timestamp, object_type, storage_dict, object_class, id_counter, threshold, check_helmet=False)` ➡️ Este método é responsável por analisar objetos detectados em um frame, associá-los a objetos previamente rastreados ou criar novas instâncias, e atualizar suas informações, como bounding boxes, timestamps e imagens extraídas. Ele também verifica a presença de capacetes em pessoas, caso solicitado. Os parâmetros são:
   Os parâmetros são:  
  - *frame*: O frame atual da análise, usado para extrair regiões de interesse (ROIs) dos objetos detectados.
  - *results*: Resultado da detecção, contendo as bounding boxes e classes dos objetos detectados no frame.
  - *current_frame*: O índice ou identificador do frame atual (por exemplo, número do frame no vídeo).
  - *timestamp*: Timestamp associado ao frame atual, indicando o momento da detecção.
  - *object_type*: O tipo de objeto a ser analisado (ex.: "pessoa" ou "veículo").
  - *storage_dict*: Dicionário usado para armazenar e rastrear objetos detectados, mapeando IDs únicos para instâncias de objetos.
  - *object_class*: A classe utilizada para criar novas instâncias de objetos (ex.: `worker` ou `vehicle`).
  - *id_counter*: Lista que mantém um contador global para atribuir IDs únicos a novos objetos detectados.
  - *threshold*: Distância máxima permitida para associar um objeto detectado a um objeto existente, com base na posição de seus centroids.
  - *check_helmet*: Booleano indicando se irá ser feita a verificação de capacete ou não no objeto
- `video_analysis(self,frame,results,current_frame,timestamp)` ➡️  Realiza uma chamada para o métdo `_object_analysis` adaptando-o para o caso em que deseja-se detectar um veículo ou uma pessoa.
- `create_obj_alert(self,current_frame,storage_dict)` ➡️  Cria alerta para o objeto, tem como parâmetros *current_frame* que indica o frame atual e *storage_dict* que dicionário usado para armazenar e rastrear objetos detectados, mapeando IDs únicos para instâncias de objetos. Remove os objetos do dicionário que não foram vistos no últioms 30 frames. No caso em que o objeto é uma pessoa verifica se a razão entre o total de frames que a pessoa foi detectada e o número de vezes que ela apareceu sem o capacete é maior que um thresholdld, caso seja o alerta é emitido. No caso em que o objeto é um veículo o alerta é emitido diretamente.
- `create_alert(self,current_frame)` ➡️ Faz chamada para `create_obj_alert()` de acordo com o tipo do objeto

## Arquivo principal
No arquivo `main.py` são criados os diretórios onde são armazendos os alertas e as imagens. Também é carregado a rede *YoloV8* com os pesos disponíveis em *model/best.pt*. São inicializadas duas instâncias da classe `videoAnalyzer`, uma para a detecção de pessoas e outra para a de veículos. É então feito o processamento do vídeo, cada um de seus frames é enviado ao modelo e então é efeito a análise do resultado obtido para gerar possíveis emissões de alertas e salvar imagens.

