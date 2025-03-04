# Importere nødvendige biblioteker
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import random
import pandas as pd
import time
from tqdm.notebook import tqdm  # For å vise fremdriftsindikatorer i Jupyter
import gc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Activation
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D, Input
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0, MobileNetV2

# ---------------------------
# GPU-OPTIMALISERING
# ---------------------------

# Forsøk å aktivere mixed precision (float16) for raskere beregning på GPU
# Fang feil i tilfelle versjonsproblemer
try:
    print("Forsøker å aktivere mixed precision trening...")
    set_global_policy('mixed_float16')
    print("Mixed precision aktivert")
    use_mixed_precision = True
except Exception as e:
    print(f"Kunne ikke aktivere mixed precision: {e}")
    print("Fortsetter med standard presisjonsformat (float32)")
    use_mixed_precision = False

# Konfigurer GPU-minnehåndtering
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        try:
            # Tillat minnevekst
            tf.config.experimental.set_memory_growth(device, True)
            print(f"Minnevekst aktivert for {device}")
        except Exception as e:
            print(f"Kunne ikke konfigurere GPU-minnevekst: {e}")
    print(f"Fant {len(physical_devices)} GPU-enheter")
else:
    print("Ingen GPU-enheter funnet")

# Forsøk å aktivere XLA-kompilering
try:
    # Aktiver XLA-kompilering for ytterligere akselerasjon
    tf.config.optimizer.set_jit(True)
    print("XLA-kompilering aktivert for raskere modellkjøring")
except Exception as e:
    print(f"Kunne ikke aktivere XLA-kompilering: {e}")
    print("Fortsetter uten XLA-optimalisering")

# ---------------------------
# KONSTANTER OG KONFIGURASJONER
# ---------------------------

# Definere bildestørrelse
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Her må du angi riktige stier til dine datafoldere
# Eksempel på stier (oppdater disse med faktiske stier)
train_pneumonia_dir = "chest_xray/train/PNEUMONIA"
train_normal_dir = "chest_xray/train/NORMAL"

# ---------------------------
# DATAFORBEREDELSE
# ---------------------------

# Funksjon for å samle alle filstier og etiketter
def prepare_cv_data():
    print("Samler filstier og etiketter...")
    pneumonia_files = [os.path.join(train_pneumonia_dir, f) for f in os.listdir(train_pneumonia_dir)]
    normal_files = [os.path.join(train_normal_dir, f) for f in os.listdir(train_normal_dir)]
    
    all_files = pneumonia_files + normal_files
    all_labels = [1] * len(pneumonia_files) + [0] * len(normal_files)
    
    # Zip sammen og shuffle
    combined = list(zip(all_files, all_labels))
    random.shuffle(combined)
    all_files, all_labels = zip(*combined)
    
    # Konverter til numpy arrays
    all_files = np.array(all_files)
    all_labels = np.array(all_labels)
    
    print(f"Totalt antall bilder: {len(all_files)}")
    print(f"Pneumonia-bilder: {np.sum(all_labels == 1)}")
    print(f"Normal-bilder: {np.sum(all_labels == 0)}")
    
    return all_files, all_labels

# Optimalisert tf.data pipeline for databehandling
def build_optimized_dataset(files, labels, is_training=False, batch_size=32):
    """
    Bygger en optimalisert tf.data.Dataset pipeline for GPU-trening
    
    Args:
        files: Liste med bildefiler
        labels: Liste med etiketter
        is_training: Om datasettet skal brukes til trening (med augmentering)
        batch_size: Størrelse på batch
    
    Returns:
        tf.data.Dataset
    """
    # Funksjon for å laste og prosessere ett bilde
    def process_path(file_path, label):
        # Last bildet
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
        img = tf.cast(img, tf.float32) / 255.0
        
        # Bildeutvidelse hvis trening
        if is_training:
            # Grunnleggende augmentering uten problematisk rotasjon
            # Horisontalt flip
            img = tf.image.random_flip_left_right(img)
            
            # Vertikalt flip (kan være nyttig avhengig av datasettet)
            if tf.random.uniform(()) > 0.8:  # 20% sjanse
                img = tf.image.flip_up_down(img)
            
            # Sentral crop med varierende størrelse (simulerer zoom)
            if tf.random.uniform(()) > 0.5:  # 50% sjanse
                # Random crop mellom 80% og 100% av bildet
                crop_factor = tf.random.uniform([], 0.8, 1.0)
                crop_height = tf.cast(tf.cast(IMG_HEIGHT, tf.float32) * crop_factor, tf.int32)
                crop_width = tf.cast(tf.cast(IMG_WIDTH, tf.float32) * crop_factor, tf.int32)
                
                # Sikre minimumsstørrelsen
                crop_height = tf.maximum(crop_height, IMG_HEIGHT // 2)
                crop_width = tf.maximum(crop_width, IMG_WIDTH // 2)
                
                # Finn sentrum for crop
                h_offset = (IMG_HEIGHT - crop_height) // 2
                w_offset = (IMG_WIDTH - crop_width) // 2
                
                # Crop og resize
                img = tf.image.crop_to_bounding_box(img, h_offset, w_offset, crop_height, crop_width)
                img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
            
            # Justere lysstyrke
            img = tf.image.random_brightness(img, 0.2)
            
            # Justere kontrast
            img = tf.image.random_contrast(img, 0.8, 1.2)
            
            # Justere metning (saturation)
            img = tf.image.random_saturation(img, 0.8, 1.2)
            
            # Justere fargetone (hue)
            img = tf.image.random_hue(img, 0.1)
            
            # Klipp verdier til [0,1]
            img = tf.clip_by_value(img, 0.0, 1.0)
        
        return img, label
    
    # Opprett tf.data.Dataset fra filer og etiketter
    file_paths_ds = tf.data.Dataset.from_tensor_slices(files)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip((file_paths_ds, labels_ds))
    
    # Shuffle med en stor buffer hvis trening
    if is_training:
        dataset = dataset.shuffle(buffer_size=min(len(files), 10000), 
                                  reshuffle_each_iteration=True)
    
    # Parallell databehandling
    dataset = dataset.map(process_path, 
                         num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch og prefetch
    dataset = dataset.batch(batch_size, drop_remainder=is_training)
    
    # Cache datasettet i minnet hvis det ikke er for stort
    if len(files) < 5000:  # Juster denne grensen basert på tilgjengelig minne
        dataset = dataset.cache()
    
    # Prefetch for å overlapp databehandling og modellutførelse
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# ---------------------------
# MODELLDEFINISJONER
# ---------------------------

# Standard CNN-modell (fra opprinnelig kode)
def create_model():
    model = Sequential()
    
    # Første konvolusjonelle blokk
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Andre konvolusjonelle blokk
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Tredje konvolusjonelle blokk
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Flatten-lag
    model.add(Flatten())
    
    # Fullt tilkoblede lag
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    # Utdatalag
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    return model

# GPU-optimalisert CNN-modell
def create_gpu_optimized_model():
    """
    Oppretter en GPU-optimalisert CNN-modell som bedre utnytter parallell prosessering
    """
    model = tf.keras.Sequential()
    
    # Første konvolusjonelle blokk - bruk av større filterstørrelser for å starte
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
    model.add(tf.keras.layers.BatchNormalization())  # Forbedrer trening og stabilitet
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    
    # Andre konvolusjonelle blokk
    model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.3))
    
    # Tredje konvolusjonelle blokk
    model.add(tf.keras.layers.Conv2D(256, (3, 3), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2D(256, (3, 3), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.4))
    
    # Global Average Pooling istedenfor Flatten - bedre ytelse og mindre parametere
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    
    # Fullt tilkoblede lag
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    
    # Utdatalag
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    return model

# GPU-optimalisert Transfer Learning-modell
def create_optimized_transfer_learning_model(base_model_type='EfficientNetB0'):
    """
    Oppretter en GPU-optimalisert transfer learning-modell
    med pre-trente vekter og effektiv struktur
    """
    # Standardinnstilling er EfficientNet som er effektiv på GPU
    if base_model_type == 'EfficientNetB0':
        base_model = tf.keras.applications.EfficientNetB0(
            weights='imagenet', 
            include_top=False, 
            input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
        )
    elif base_model_type == 'MobileNetV2':
        base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet', 
            include_top=False, 
            input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
        )
    elif base_model_type == 'VGG16':
        base_model = tf.keras.applications.VGG16(
            weights='imagenet', 
            include_top=False, 
            input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
        )
    else:
        raise ValueError("Støttede modeller: 'EfficientNetB0', 'MobileNetV2', 'VGG16'")
    
    # Fryse de pre-trente lagene
    base_model.trainable = False
    
    # Bruk Functional API for å bygge modellen
    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    
    # Preprosessering basert på valgt modell
    if base_model_type == 'EfficientNetB0':
        x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    elif base_model_type == 'MobileNetV2':
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    elif base_model_type == 'VGG16':
        x = tf.keras.applications.vgg16.preprocess_input(inputs)
    
    # Få aktiveringene fra base-modellen
    x = base_model(x, training=False)
    
    # Legg til klassifikasjonshode
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    # Skap fullstendig modell
    model = tf.keras.Model(inputs, outputs)
    
    return model

# ---------------------------
# KRYSSVALIDERING
# ---------------------------

def perform_optimized_cross_validation(n_splits=5, epochs=20, batch_size=32, model_func=create_gpu_optimized_model):
    """
    Utfører GPU-optimalisert k-fold kryssvalidering på pneumonia-datasettet
    
    Args:
        n_splits: Antall folds for kryssvalidering
        epochs: Antall epoker for trening
        batch_size: Batchstørrelse for trening (øk for å utnytte GPU bedre)
        model_func: Funksjon som oppretter modellen
    
    Returns:
        Dictionary med resultater
    """
    # Initialisere kryssvalidering
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Forberede resultatlagring
    fold_val_accuracies = []
    fold_val_losses = []
    fold_val_precisions = []
    fold_val_recalls = []
    fold_val_aucs = []
    results_df = pd.DataFrame(columns=[
        'fold', 'val_accuracy', 'val_loss', 'val_precision', 'val_recall', 'val_auc',
        'train_samples', 'val_samples', 'pneumonia_train', 'normal_train',
        'pneumonia_val', 'normal_val', 'class_weight_normal', 'class_weight_pneumonia',
        'training_time'
    ])
    
    # For å følge treningsprosessen
    main_log_dir = os.path.join("logs", "pneumonia_cv_gpu_optimized")
    if not os.path.exists(main_log_dir):
        os.makedirs(main_log_dir)
    
    # Utfør kryssvalidering
    print(f"Starter GPU-optimalisert {n_splits}-fold kryssvalidering...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_files, all_labels)):
        start_time = time.time()
        print(f"\n{'-'*50}")
        print(f"Trener FOLD {fold+1}/{n_splits}")
        print(f"{'-'*50}")
        
        # Dele dataene for denne folden
        train_files, val_files = all_files[train_idx], all_files[val_idx]
        train_labels, val_labels = all_labels[train_idx], all_labels[val_idx]
        
        # Beregne klassevekter for denne folden
        train_pneumonia_count = np.sum(train_labels == 1)
        train_normal_count = np.sum(train_labels == 0)
        
        weight_for_0 = (1 / train_normal_count) * (len(train_labels) / 2.0)
        weight_for_1 = (1 / train_pneumonia_count) * (len(train_labels) / 2.0)
        
        class_weights = {0: weight_for_0, 1: weight_for_1}
        
        # Bruk den optimaliserte tf.data pipelinen
        train_dataset = build_optimized_dataset(
            train_files, train_labels, is_training=True, batch_size=batch_size
        )
        
        val_dataset = build_optimized_dataset(
            val_files, val_labels, is_training=False, batch_size=batch_size
        )
        
        # Dataset-info for loggføring
        train_pneumonia = np.sum(train_labels == 1)
        train_normal = np.sum(train_labels == 0)
        val_pneumonia = np.sum(val_labels == 1)
        val_normal = np.sum(val_labels == 0)
        
        print(f"Treningsdatasett: {len(train_files)} bilder "
              f"({train_pneumonia} pneumonia, {train_normal} normal)")
        print(f"Valideringsdatasett: {len(val_files)} bilder "
              f"({val_pneumonia} pneumonia, {val_normal} normal)")
        print(f"Klassevekter: Normal={class_weights[0]:.4f}, Pneumonia={class_weights[1]:.4f}")
        
        # Beregn steps_per_epoch for trening
        steps_per_epoch = len(train_files) // batch_size
        validation_steps = len(val_files) // batch_size
        
        if steps_per_epoch == 0:
            steps_per_epoch = 1
        if validation_steps == 0:
            validation_steps = 1
            
        # Opprette og kompilere modellen for denne folden
        with tf.device('/gpu:0'):  # Eksplisitt plassering på GPU
            model = model_func()
            
            # Optimizer med tilpasset læringsrate
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            
            # For mixed precision, bruk loss scaling
            if use_mixed_precision:
                try:
                    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
                    print("Loss scale optimizer aktivert for mixed precision")
                except Exception as e:
                    print(f"Kunne ikke konfigurere loss scale optimizer: {e}")
                    print("Fortsetter med standard optimizer")
            
            model.compile(
                loss='binary_crossentropy',
                optimizer=optimizer,
                metrics=[
                    'accuracy',
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall(),
                    tf.keras.metrics.AUC()
                ]
            )
        
        # Definere callbacks for denne folden
        fold_log_dir = os.path.join(main_log_dir, f"fold_{fold+1}")
        if not os.path.exists(fold_log_dir):
            os.makedirs(fold_log_dir)
        
        callbacks = [
            tf.keras.callbacks.TensorBoard(
                log_dir=fold_log_dir,
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch',
                profile_batch=0  # Deaktiver profiling for ytelse
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(fold_log_dir, 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=8,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=4,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Trene modellen
        print(f"Trener modell for fold {fold+1}...")
        
        # Mål treningstid
        train_start = time.time()
        
        history = model.fit(
            train_dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_dataset,
            validation_steps=validation_steps,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        train_time = time.time() - train_start
        print(f"Treningstid: {train_time:.2f} sekunder")
        
        # Evaluere modellen på valideringssettet
        print(f"Evaluerer modell for fold {fold+1}...")
        val_results = model.evaluate(
            val_dataset,
            steps=validation_steps,
            verbose=1
        )
        
        # Lagre resultater
        fold_val_loss = val_results[0]
        fold_val_acc = val_results[1]
        fold_val_precision = val_results[2]
        fold_val_recall = val_results[3]
        fold_val_auc = val_results[4]
        
        fold_val_losses.append(fold_val_loss)
        fold_val_accuracies.append(fold_val_acc)
        fold_val_precisions.append(fold_val_precision)
        fold_val_recalls.append(fold_val_recall)
        fold_val_aucs.append(fold_val_auc)
        
        # Lagre til dataframe
        results_df = results_df._append({
            'fold': fold+1,
            'val_accuracy': fold_val_acc,
            'val_loss': fold_val_loss,
            'val_precision': fold_val_precision,
            'val_recall': fold_val_recall,
            'val_auc': fold_val_auc,
            'train_samples': len(train_files),
            'val_samples': len(val_files),
            'pneumonia_train': train_pneumonia,
            'normal_train': train_normal,
            'pneumonia_val': val_pneumonia,
            'normal_val': val_normal,
            'class_weight_normal': class_weights[0],
            'class_weight_pneumonia': class_weights[1],
            'training_time': train_time
        }, ignore_index=True)
        
        # Plott læringshistorikk for denne folden
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title(f'Fold {fold+1} Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'Fold {fold+1} Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(fold_log_dir, 'training_history.png'))
        plt.show()
        
        fold_time = time.time() - start_time
        print(f"Fold {fold+1} fullført på {fold_time:.2f} sekunder")
        print(f"Validering - Nøyaktighet: {fold_val_acc:.4f}, Tap: {fold_val_loss:.4f}")
        
        # Rydde minne mellom folds
        tf.keras.backend.clear_session()
        
        # Frigjør GPU-minne eksplisitt
        gc.collect()
    
    # Beregne og vise gjennomsnittsresultater
    mean_val_loss = np.mean(fold_val_losses)
    mean_val_acc = np.mean(fold_val_accuracies)
    mean_val_precision = np.mean(fold_val_precisions)
    mean_val_recall = np.mean(fold_val_recalls)
    mean_val_auc = np.mean(fold_val_aucs)
    
    std_val_loss = np.std(fold_val_losses)
    std_val_acc = np.std(fold_val_accuracies)
    std_val_precision = np.std(fold_val_precisions)
    std_val_recall = np.std(fold_val_recalls)
    std_val_auc = np.std(fold_val_aucs)
    
    print("\n" + "="*50)
    print("Kryssvalideringsresultater:")
    print("="*50)
    print(f"Gjennomsnittlig validering nøyaktighet: {mean_val_acc:.4f} ± {std_val_acc:.4f}")
    print(f"Gjennomsnittlig validering tap: {mean_val_loss:.4f} ± {std_val_loss:.4f}")
    print(f"Gjennomsnittlig validering presisjon: {mean_val_precision:.4f} ± {std_val_precision:.4f}")
    print(f"Gjennomsnittlig validering recall: {mean_val_recall:.4f} ± {std_val_recall:.4f}")
    print(f"Gjennomsnittlig validering AUC: {mean_val_auc:.4f} ± {std_val_auc:.4f}")
    
    # Lagre resultater til fil
    results_df.to_csv(os.path.join(main_log_dir, 'fold_results.csv'), index=False)
    
    # Plotte resultatene på tvers av folds
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.bar(range(1, n_splits+1), fold_val_accuracies)
    plt.axhline(y=mean_val_acc, color='r', linestyle='-', label=f'Mean: {mean_val_acc:.4f}')
    plt.title('Validation Accuracy by Fold')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.bar(range(1, n_splits+1), fold_val_losses)
    plt.axhline(y=mean_val_loss, color='r', linestyle='-', label=f'Mean: {mean_val_loss:.4f}')
    plt.title('Validation Loss by Fold')
    plt.xlabel('Fold')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.bar(range(1, n_splits+1), fold_val_precisions)
    plt.axhline(y=mean_val_precision, color='r', linestyle='-', label=f'Mean: {mean_val_precision:.4f}')
    plt.title('Validation Precision by Fold')
    plt.xlabel('Fold')
    plt.ylabel('Precision')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.bar(range(1, n_splits+1), fold_val_recalls)
    plt.axhline(y=mean_val_recall, color='r', linestyle='-', label=f'Mean: {mean_val_recall:.4f}')
    plt.title('Validation Recall by Fold')
    plt.xlabel('Fold')
    plt.ylabel('Recall')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(main_log_dir, 'cross_validation_results.png'))
    plt.show()
    
    # Samle resultater i en dictionary
    results_summary = {
        'mean_val_accuracy': mean_val_acc,
        'std_val_accuracy': std_val_acc,
        'mean_val_loss': mean_val_loss,
        'std_val_loss': std_val_loss,
        'mean_val_precision': mean_val_precision,
        'std_val_precision': std_val_precision,
        'mean_val_recall': mean_val_recall,
        'std_val_recall': std_val_recall,
        'mean_val_auc': mean_val_auc,
        'std_val_auc': std_val_auc,
        'num_folds': n_splits,
        'epochs': epochs,
        'batch_size': batch_size,
        'avg_training_time_per_fold': results_df['training_time'].mean()
    }
    
    return results_summary, results_df, model

# ---------------------------
# HOVEDPROGRAM
# ---------------------------

if __name__ == "__main__":
    # 1. Last inn datasettet
    all_files, all_labels = prepare_cv_data()
    
    # 2. Definere hyperparametere for trening
    batch_size = 64  # Øk denne til 128 hvis GPU-en har nok minne
    epochs = 20
    n_splits = 5
    
    # 3. Kjør den optimaliserte kryssvalideringen
    print("Starter GPU-optimalisert kryssvalidering...")
    start_time = time.time()
    
    # Velg modell: create_model, create_gpu_optimized_model, eller create_optimized_transfer_learning_model
    cv_results, cv_df, final_model = perform_optimized_cross_validation(
        n_splits=n_splits,
        epochs=epochs,
        batch_size=batch_size,
        model_func=create_gpu_optimized_model  # Velg din optimaliserte modell
    )
    
    total_time = time.time() - start_time
    print(f"Total kjøretid: {total_time:.2f} sekunder")
    
    # 4. Vis resultater
    print("\nResultater med GPU-optimalisert modell:")
    for key, value in cv_results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # 5. Lagre modellen
    final_model.save('pneumonia_model_gpu_optimized.h5')
    
    # 6. Sammenlign med andre modeller
    # Valgfritt: Kjør kryssvalidering med ulike modeller for å sammenligne ytelse
    """
    print("\nTester transfer learning-modell med EfficientNetB0...")
    cv_results_efficient, cv_df_efficient, _ = perform_optimized_cross_validation(
        n_splits=3,  # Bruk færre folds for raskere eksperimentering
        epochs=epochs,
        batch_size=batch_size,
        model_func=lambda: create_optimized_transfer_learning_model('EfficientNetB0')
    )
    
    print("\nTester transfer learning-modell med MobileNetV2...")
    cv_results_mobile, cv_df_mobile, _ = perform_optimized_cross_validation(
        n_splits=3,
        epochs=epochs,
        batch_size=batch_size,
        model_func=lambda: create_optimized_transfer_learning_model('MobileNetV2')
    )
    """