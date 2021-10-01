##########################################################
## Set Up ##
##########################################################

cat("\014") # clear console
rm(list=ls()) # clear workspace
gc # garbage collector

require(pacman)
p_load(metafor, dplyr, tinytex, knitr, kableExtra, DescTools, reshape2, metaSEM, lavaan, semPlot, psych, 
       ggplot2, keras, tensorflow, glmnet, caret, ModelMetrics)

set.seed(52) # set overall random seed for reproducibility

##########################################################
## Load Data ##
##########################################################

dtRaw <- haven::read_spss("M:/.00 Research/Snowball/PsyCorona-DataCleaning/data/collab data/c19 collab Data/RMD61 Thesis 2020/RMD61_Agostini_2021-09-29 19-25 UTC.sav")

## Prep Data ##
dtRaw[dtRaw == -99] <- NA

##########################################################
## Make Relevant Vars
##########################################################

# make a relevant dataframe
  tmp <- dtRaw %>%
    select(w21_affAnx, w21_affBor, w21_affEnerg, w21_affAng, w21_affLov, w21_c19IShould,
           w22_VaccineYesNo)

# clean relevant dataframe
  tmp$w22_VaccineYesNo[tmp$w22_VaccineYesNo == 2] <- NA
  tmp$miss <- rowSums(is.na(tmp))

  dt <- tmp %>%
    filter(miss == 0) %>%
    select(-miss)

  rm(tmp)

# training data
  train_data <- list(X = dt[1:round((nrow(dt)*4/5)),1:6],
                     Y = dt[1:round((nrow(dt)*4/5)),7])
  test_data <- list(X = dt[(round((nrow(dt)*4/5)+1)):nrow(dt),1:6],
                    Y = dt[((round(nrow(dt)*4/5)+1)):nrow(dt),7])

##########################################################
## Single-layer FNN
##########################################################
# construct model
  fnn <- keras_model_sequential() 
  fnn %>% 
    layer_dense(units = 6, activation = 'relu', input_shape = c(6)) %>% 
    layer_dense(units = 1, activation = 'sigmoid')
  
  fnn %>% compile(
    loss = 'binary_crossentropy',
    optimizer = optimizer_adam(),
    metrics = c('accuracy')
  )
  
# fit model
  history = fnn %>% fit(
    as.matrix(train_data$X),
    as.matrix(train_data$Y), 
    epochs = 70,
    batch_size = 128)

plot(history)
  
fnn %>% evaluate(as.matrix(test_data$X), as.matrix(test_data$Y))

probs = predict(fnn, as.matrix(test_data$X))
preds = fnn %>% predict(as.matrix(test_data$X)) %>% `>`(0.5) %>% k_cast("int32") %>% as.numeric

caret::confusionMatrix(as.factor(preds), as.factor(test_data$Y), positive = "1")





  