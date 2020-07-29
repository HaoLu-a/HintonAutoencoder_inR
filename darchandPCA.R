data(iris)

library(stats)
library(darch)
library(ggplot2)
library(reshape2)
library(futile.logger)
library(lattice)
library(caret)
library(Rcpp)
library(ggplot2)
library(ggbiplot)


darchiris <- darch(iris[,1:4], iris[,1:4], c(4,8,2,8,4), 
                   darch.isClass = F,
               preProc.params = list(method = c("center", "scale")), 
               darch.numEpochs = 20, darch.batchSize = 6,
               darch.unitFunction = "tanhUnit",
               darch.fineTuneFunction = "minimizeAutoencoder")







a <- predict(darchiris,inputLayer = 1,type = "raw", outputLayer = 3, newdata = iris[,1:4])

ad<- data.frame(a)

p <- ggplot(ad,aes(x = X1, y = X2, color = iris[,5])) +
  geom_point(shape=19) +stat_ellipse(aes(x=X1, y=X2, color = iris[,5], group = iris[,5]),type = "norm")
print(p)


pcairis <- prcomp(iris[,1:4], center = TRUE, scale = F)
g <- ggbiplot(pcairis, obs.scale = 1, var.scale = 1, 
              groups = iris[,5], ellipse = TRUE, var.axes = F, 
              circle = TRUE,varname.size = 3, varname.adjust = 1.5, 
              varname.abbrev = FALSE)
#g <- g + scale_color_discrete(name = '')
g <- g + theme(legend.direction = 'horizontal', 
               legend.position = 'top')
print(g)

par(mfrow = c(2,1))
print(p)
print(g)
