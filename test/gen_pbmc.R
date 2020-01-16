library(harmony)

metadata <- read.table("data/10x_pbmc/metadata.csv", header = TRUE, sep = ',')
X <- read.table("data/10x_pbmc/pca.txt")

start <- Sys.time()
Z <- HarmonyMatrix(X, metadata, "Channel", do_pca = FALSE)
end <- Sys.time()

print(end - start)

write.table(Z, file = "result/pbmc_harmony_z.txt", quote = FALSE, row.names = FALSE, col.names = FALSE)