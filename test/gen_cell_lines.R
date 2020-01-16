library(harmony)

metadata <- read.table("data/cell_lines/metadata.csv", header = TRUE, sep = ',')
X <- read.table("data/cell_lines/pca.txt")

start <- Sys.time()
Z <- HarmonyMatrix(X, metadata, "dataset", do_pca = FALSE)
end <- Sys.time()

print(end - start)

write.table(Z, file = "result/cell_lines_harmony_z.txt", quote = FALSE, row.names = FALSE, col.names = FALSE)