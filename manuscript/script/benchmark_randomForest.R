library(randomForest)
library(pROC)
library(PRROC)
library(RColorBrewer)

df_gene <- read.table('../data/TCGA_COAD_gene.txt', sep = '\t', quote = '', header = T, stringsAsFactors = F)
df_microbe <- read.table('../data/TCGA_COAD_microbe.txt', sep = '\t', quote = '', header = T, stringsAsFactors = F)
df_meta <- read.table('../data/COAD_sample_metadata.processed.txt', sep = '\t', quote = '', header = T, stringsAsFactors = F)

df_feature_matrix <- cbind(df_gene[, 2:ncol(df_gene)], df_microbe[, 2:ncol(df_microbe)])
df_output_matrix <- df_meta$sample_type2
df_microbe_matrix_filtered <- as.data.frame(df_feature_matrix)
df_meta_vector <- as.factor(df_output_matrix)

set.seed(1234)
training_index <- sort(sample(1:nrow(df_microbe_matrix_filtered), size = round(0.7 * nrow(df_microbe_matrix_filtered))))

# If the training/test dataset does not contain all the class, add one sample from the other dataset
df_test_sample <- as.data.frame(table(df_meta_vector[-training_index]))
for (k in 1:nrow(df_test_sample)) {
    if (df_test_sample[k, 2] == 0) {
        class <- df_test_sample[k, 1]
        list_index <- which(df_meta_vector == class)
        training_index <- training_index[which(training_index != list_index[1])]
    }
}
df_train_sample <- as.data.frame(table(df_meta_vector[training_index]))
for (k in 1:nrow(df_train_sample)) {
    if (df_train_sample[k, 2] == 0) {
        class <- df_train_sample[k, 1]
        list_index <- which(df_meta_vector == class)
        training_index <- append(training_index, list_index[1])
    }
}

x_train <- df_microbe_matrix_filtered[training_index, ]
x_test <- df_microbe_matrix_filtered[-training_index, ]
y_train <- df_meta_vector[training_index]
y_test <- df_meta_vector[-training_index]

for (ntree in c(50, 100, 200, 250, 500, 750, 1000, 1500, 2000)) {
    rf_model <- randomForest(x = x_train, y = y_train, ntree = ntree)
    pred_test <- predict(rf_model, newdata = x_test)
    pred_train_prob <- predict(rf_model, newdata = x_train, type = "prob")
    pred_test_prob <- predict(rf_model, newdata = x_test, type = "prob")

    roc_multi_train <- multiclass.roc(y_train, pred_train_prob)
    auc_multi_train <- auc(roc_multi_train)
    roc_multi_test <- multiclass.roc(y_test, pred_test_prob)
    auc_multi_test <- auc(roc_multi_test)

    train_acc <- as.numeric(1 - rf_model$err.rate[nrow(rf_model$err.rate), "OOB"])
    test_acc <- mean(pred_test == y_test)
    print(ntree)
    print(test_acc)
    print(auc_multi_test)
}