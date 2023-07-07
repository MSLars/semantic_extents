# local bert_model = "roberta-large";
local bert_model = "roberta-base";
local bert_dim = 1024;

local learn_rate = 1e-4;
{
    dataset_reader : {
        // This name needs to match the name that you used to register your dataset reader, with
        // the call to `@DatasetReader.register()`.
        type: "sequence",
        // These other parameters exactly match the constructor parameters of your dataset reader class.
    },

    train_data_path: "/home/lars/Projects/ace_preprocessing/preprocessed_relation_classification_data/train.jsonl",
    validation_data_path: "/home/lars/Projects/ace_preprocessing/preprocessed_relation_classification_data/dev.jsonl",
    test_data_path: "/home/lars/Projects/ace_preprocessing/preprocessed_relation_classification_data/test.jsonl",
    evaluate_on_test: true,
    model: {
        type: "sequence_classification",
        transformer_model: bert_model,
    },

    data_loader: {
        batch_sampler:{
            batch_size: 32,
            type: 'bucket',
        },
    },

    trainer: {
        validation_metric: "+f1-macro-overall",
        num_epochs: 50,
        patience: 5,
        cuda_device: 0,
        optimizer: {
            type: "huggingface_adamw",
            lr: learn_rate,
        },
    }
}
