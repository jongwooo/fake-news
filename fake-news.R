library(tm)
library(SnowballC)
library(lsa)
library(caret)

# Read csv file
fake.news.df<-read.csv("data/FakeNews.csv")

# Remove rows with missing values
fake.news.df<-fake.news.df[complete.cases(fake.news.df),]

# Remove non-specific data
fake.news.df<-fake.news.df[fake.news.df$real %in% c(1, 0),]

# Calculate the number of rows
row_size<-nrow(fake.news.df)

# Create corpus from data frame
corpus<-VCorpus(VectorSource(fake.news.df[1:row_size, 1]))

# Create label
label<-as.factor(fake.news.df[1:row_size, 5])

# Tokenization
corpus<-tm_map(corpus, stripWhitespace)
corpus<-tm_map(corpus, removePunctuation)
corpus<-tm_map(corpus, removeNumbers)

# Stopwords
corpus<-tm_map(corpus, removeWords, stopwords("english"))

# Stemming
corpus<-tm_map(corpus, stemDocument)

# Compute TF-IDF
tdm<-TermDocumentMatrix(corpus)

if (any(apply(tdm, 2, sum)==0)) {
  tdm<-tdm[, apply(tdm, 2, sum)!=0]
}

tridf<-weightTfIdf(tdm)

# Extract (20) concepts
lsa.tfidf<-lsa(tridf, dims=20)

# Convert to data frame
words.df<-as.data.frame(as.matrix(lsa.tfidf$dk))

# Set seed
set.seed(123)

# Sample 60% of the data for training
training<-sample(row_size, 0.6*row_size)

# Run logistic model on training
trainData<-cbind(label=label[training], words.df[training,])
reg<-glm(label ~ ., data=trainData, family=binomial)

# Compute accuracy on validation set
validData<-cbind(label=label[-training], words.df[-training,])
pred<-predict(reg, newdata=validData, type="response")

# Produce confusion matrix
confusionMatrix(table(ifelse(pred>0.5, 1, 0), validData$label))
