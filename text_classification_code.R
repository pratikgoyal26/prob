rm(list=ls(all=TRUE))
library(SnowballC)
library(tm)
library(wordcloud)
library(caret)
library(C50)
library(randomForest)
library(ggplot2)
library(RWeka)

set.seed(142)

ticketsData <- read.csv('TextClassificationData.csv', header = TRUE, stringsAsFactors = FALSE)

## Clear unwanted fieldss
ticketsData$previous_appointment <- NULL
ticketsData$ID <- NULL
ticketsData$fileid <- NULL
ticketsData$categories <- NULL

## manage the target fiel
ticketsData$sub_categories <- tolower(ticketsData$sub_categories)
ticketsData$sub_categories <- factor(ticketsData$sub_categories)
summary(ticketsData$sub_categories)

## Extract ticketsData from RTF
ticketsData$desc <- gsub("[^a-zA-Z]", " ", ticketsData$DATA)

## combine the fields into 1 field
ticketsData$description <- paste(ticketsData$SUMMARY, ticketsData$desc, sep = " ")

## remove other fields
ticketsData$SUMMARY <- NULL
ticketsData$ticketsData <- NULL
ticketsData$desc <- NULL

## Declare the stopwords to remove from ticketsCorpus
rtfText = c("rtf", "ansi", "ftnbj", "fonttbl", "fswiss", "arial",
            "colortbl", "red", "green", "blue", "stylesheet", "normal",
            "par", "formshade", "paragraph", "font", "margl", "margr", "margt",
            "margb", "headery", "footery", "sectd", "plain", "f", "fs", "cf", "cb", "cs",
            "pard", "sbkpage", "pgncont", "sscharaux", "pm", "b", "am", "additive",
            "marglsxn", "margrsxn", "margtsxn", "margbsxn", "xxxx", "=")

## Create the ticketsCorpus
ticketsCorpus <- Corpus(VectorSource(ticketsData$description))

## ticketsData preprocessing
ticketsCorpus <- tm_map(ticketsCorpus,content_transformer(tolower))
ticketsCorpus <- tm_map(ticketsCorpus, removePunctuation) 
ticketsCorpus <- tm_map(ticketsCorpus, removeNumbers)
ticketsCorpus <- tm_map(ticketsCorpus, removeWords, stopwords("english"))
ticketsCorpus <- tm_map(ticketsCorpus, removeWords, rtfText)   
ticketsCorpus <- tm_map(ticketsCorpus, stripWhitespace)

## Stemming
ticketsCorpus <- tm_map(ticketsCorpus, stemDocument, language = "english")  

## inspect the text
writeLines(as.character(ticketsCorpus[[5317]]))

## Declare the list of subcategories will be used for Recall
subCategory <- c('cancellation',
                 'change of hospital',
                 'change of pharmacy',
                 'change of provider',
                 'follow up on previous request',
                 'junk',
                 'lab results',
                 'medication related',
                 'new appointment',
                 'others',
                 'prior authorization',
                 'provider',
                 'queries from insurance firm',
                 'queries from pharmacy',
                 'query on current appointment',
                 'refill',
                 'rescheduling',
                 'running late to appointment',
                 'sharing of health records (fax, e-mail, etc.)',
                 'sharing of lab records (fax, e-mail, etc.)',
                 'symptoms'
)

## declare the function 

getResults <- function (textDtm){

  ticketsDTMdf <- data.frame(as.matrix(textDtm)) #Using sparse terms
  
  ticketsTrainDTM = ticketsDTMdf[sample_index,] #dtm 
  ticketsTestDTM = ticketsDTMdf[-sample_index,] #dtm 
  
  #### Decision Tree ####
  dtc50Model <- C5.0(ticketsTrainDTM, ticketsTrainData$sub_categories)
  
  ### Predict on Test ticketsData
  dtc50TestPred <- predict(dtc50Model, ticketsTestDTM)
  dtc50TestConf <- table(ticketsTextData$sub_categories, dtc50TestPred)
  dtc50TestAcc <- sum(diag(dtc50TestConf))/nrow(ticketsTestDTM) * 100
  
  print(paste("The Accuracy for Decision Trees is", dtc50TestAcc))
  
  ## Print confusion matrix
  dtc50Confusion <- as.data.frame(dtc50TestConf)
  
  dtc50Plot <- ggplot(dtc50Confusion)
  dtc50Plot + geom_tile(aes(x=Var1, y=dtc50TestPred, fill=Freq)) + 
    scale_x_discrete(name="Actual Class") + scale_y_discrete(name="Predicted Class") + 
    scale_fill_gradient(breaks=seq(from=-.5, to=4, by=.2)) + 
    labs(fill="Normalized\nFrequency")
  
  
  ### Random Forest
  RfModel <- randomForest(ticketsTrainDTM, ticketsTrainData$sub_categories, ntree = 30)
  ### Predict on Test ticketsData
  rfTestPred <- predict(RfModel, ticketsTestDTM)
  rfTestConf <-table(ticketsTextData$sub_categories, rfTestPred)
  rfTestAcc <- sum(diag(rfTestConf))/nrow(ticketsTestDTM) * 100
  
  print(paste("The Accuracy for Random Forest is", rfTestAcc))
  
  ## Print confusion matrix
  rfConfusion <- as.data.frame(as.table(rfTestConf))
  
  
  rfPlot <- ggplot(rfConfusion)
  rfPlot + geom_tile(aes(x=Var1, y=rfTestPred, fill=Freq)) + 
    scale_x_discrete(name="Actual Class") + scale_y_discrete(name="Predicted Class") + 
    scale_fill_gradient(breaks=seq(from=-.5, to=4, by=.2)) + 
    labs(fill="Normalized\nFrequency")
  
  
  ## print all the ticketsData in excel to figure which records were classified incorrectly
  results <- ticketsTextData
  results$result <- rfTestPred
  
  allresult <- results
  for (i in 1:nrow(results)){
    i <- 2
    if (results[i,1] == results[i,3]){
      allresult[i,] <- NA
    }
  }
  allresult <- na.omit(allresult)
  
  write.csv(allresult,'textticketsDataresult.csv')
  
  
  ### Print the recall for each subcategory
    
  for (i in subCategory){
    DesiredClass = which(colnames(dtc50TestConf)==i)
    
    dtc50RecallValue = round((dtc50TestConf[DesiredClass,DesiredClass]/sum(dtc50TestConf[DesiredClass,]))*100,2)
    print(paste("The recall for Decision Trees ",i," is", dtc50RecallValue))
    
    rfRecallValue = round((rfTestConf[DesiredClass,DesiredClass]/sum(rfTestConf[DesiredClass,]))*100,2)
    print(paste("The recall for Random Forest ",i," is", rfRecallValue))

  }
  
}

## Creating Document term Matrix
ticketsDTM <- DocumentTermMatrix(ticketsCorpus)
ticketsDTM

## Remove sparse terms 
ticketsSparseDTM <- removeSparseTerms(ticketsDTM, 0.95) 
ticketsSparseDTM

## Check Frequent terms
termFrequency <- colSums(as.matrix(ticketsSparseDTM))   

## Create wordcloud
wordcloud(names(termFrequency), termFrequency, scale = c(6, .1), colors = brewer.pal(6, 'Dark2'))

#split ticketsData into train and test
sample_index <- createDataPartition(ticketsData$sub_categories, p=0.75, list=FALSE)

# creating training and test Datasets
ticketsTrainData <- ticketsData[sample_index, ]
ticketsTextData <- ticketsData[-sample_index, ]

table(ticketsTrainData$sub_categories)
table(ticketsTextData$sub_categories)

getResults(ticketsSparseDTM)
# Accuracy for Decision Tree : 61.57
# Accuracy for Random Forest : 66.29

### Found Random Forest as the best model with sparsity .95


#### Other trials made ####
### Use Bi-grams
BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2)) # create n-grams
ticketsBiDTM <- DocumentTermMatrix(ticketsCorpus, control = list(tokenize = BigramTokenizer)) # create tdm from n-grams
ticketsBiDTM

## Remove sparse terms 
ticketsSparseBiDTM <- removeSparseTerms(ticketsBiDTM, 0.95) 
ticketsSparseBiDTM

getResults(ticketsSparseBiDTM)
# Accuracy for Decision Tree with Bigrams : 52.33
# Accuracy for Random Forest with Bigrams : 55.30

### Use SVD

termSvd <- svd(ticketsSparseDTM)
s <- diag(termSvd[[1]])
u <- termSvd[[2]]
v <- termSvd[[3]]

eigenval = termSvd$d
e_sqare_energy = (eigenval/sum(eigenval))*100
cumsum(e_sqare_energy)

vt <- t(v)
s=s[1:107,1:107]
u=u[,1:107]
vt=vt[1:107,]

ticketsSvdDtm=u%*%s%*%vt
ticketsSvdDtm

getResults(ticketsSvdDtm)
# Accuracy for Decision Tree using SVD : 58.17
# Accuracy for Random Forest using SVD : 64.67

## Tf-idf
# Dictionary creation
ticketsDict <- c(findFreqTerms(ticketsDTM, 10))

# Min frequency 5 words are chosen and TFIDF is considered
# # TFIDF
ticketsTFIDFDtm <-DocumentTermMatrix(ticketsCorpus,control=list(weighting=weightTfIdf, dictionary = ticketsDict))

## Remove sparse terms 
ticketsSparseTfDTM <- removeSparseTerms(ticketsTFIDFDtm, 0.95) 
ticketsSparseTfDTM

getResults(ticketsSparseTfDTM)
# Accuracy for Decision Tree using TF-IDF : 60.87
# Accuracy for Random Forest using TF-IDF : 66.02
