#!/usr/bin/env Rscript
# icenreg.R - 区间删失数据生存分析和AFT模型
# 用法: Rscript icenreg.R intervals.csv mid.csv low.csv up.csv [bootstrap_flag]

cat("icenreg.R started\n")
cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"), "\n")

# -------- 自动安装并加载必要的包 --------
required_packages <- c("survival", "icenReg", "flexsurv", "mclust")
missing_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]

# 安装缺少的包
if(length(missing_packages) > 0) {
  cat("Installing missing packages:", paste(missing_packages, collapse=", "), "\n")
  install.packages(missing_packages, repos = "https://cloud.r-project.org")
}

# 尝试加载包并检查是否成功
load_package <- function(pkg) {
  status <- tryCatch({
    library(pkg, character.only = TRUE)
    TRUE
  }, error = function(e) {
    cat("Failed to load package:", pkg, "\n")
    cat("Error:", conditionMessage(e), "\n")
    FALSE
  })
  return(status)
}

pkg_status <- sapply(required_packages, load_package)
if(!all(pkg_status)) {
  cat("Warning: Some required packages could not be loaded.\n")
  cat("Results may be limited or unavailable.\n")
}

# -------- 读取参数 --------
args <- commandArgs(trailingOnly = TRUE)
intervals_file <- args[1]
mid_file <- if(length(args) >= 2 && args[2] != "NA") args[2] else NULL
low_file <- if(length(args) >= 3 && args[3] != "NA") args[3] else NULL
up_file <- if(length(args) >= 4 && args[4] != "NA") args[4] else NULL
bootstrap_flag <- if(length(args) >= 5) as.logical(as.integer(args[5])) else FALSE
out_dir <- dirname(intervals_file)

# -------- 读取数据 --------
intervals <- tryCatch({
  read.csv(intervals_file, stringsAsFactors = FALSE)
}, error = function(e) {
  cat("Error reading intervals file:", conditionMessage(e), "\n")
  NULL
})

if(is.null(intervals)) {
  cat("Failed to read intervals data. Exiting.\n")
  quit(status = 1)
}

# -------- 拟合 mid/low/up 的 AFT 模型 --------
fit_survreg <- function(file, name) {
  if(is.null(file)) return(NULL)
  
  cat("Fitting survreg on", name, "...\n")
  df <- tryCatch(read.csv(file, stringsAsFactors = FALSE), 
                error = function(e) {
                  cat("Error reading", name, "file:", conditionMessage(e), "\n")
                  return(NULL)
                })
  
  if(is.null(df)) return(NULL)
  
  # 尝试拟合对数正态AFT
  fit <- tryCatch(
    survival::survreg(Surv(T, event) ~ BMI, data = df, dist = "lognormal"),
    error = function(e) {
      cat("survreg(lognormal) failed for", name, "-> trying weibull; error:\n")
      print(e)
      # 尝试威布尔AFT作为后备
      tryCatch(
        survival::survreg(Surv(T, event) ~ BMI, data = df, dist = "weibull"),
        error = function(e2) {
          cat("survreg fallback also failed for", name, "\n")
          NULL
        }
      )
    }
  )
  
  if(!is.null(fit)) {
    # 修复：确保所有向量具有相同的长度
    coef_names <- names(coef(fit))
    coef_values <- coef(fit)
    
    # 获取标准误差，确保与系数长度匹配
    se <- tryCatch({
      sqrt(diag(fit$var))
    }, error = function(e) {
      cat("Error extracting std.errors, using NAs:", conditionMessage(e), "\n")
      rep(NA, length(coef_values))
    })
    
    # 如果长度不匹配，进行扩展
    if(length(se) != length(coef_values)) {
      cat("Warning: std.error length mismatch, padding with NA\n")
      if(length(se) < length(coef_values)) {
        se <- c(se, rep(NA, length(coef_values) - length(se)))
      } else {
        se <- se[1:length(coef_values)]
      }
    }
    
    # 创建数据框，确保所有向量长度相同
    dist_vec <- rep(fit$dist, length(coef_values))
    coefs <- data.frame(
      term = coef_names,
      estimate = coef_values,
      std.error = se,
      distribution = dist_vec,
      stringsAsFactors = FALSE
    )
    
    write.csv(coefs, file.path(out_dir, paste0("aft_", name, "_coef.csv")), row.names = FALSE)
    cat("Successfully wrote coefficients for", name, "\n")
  }
  
  return(fit)
}

# 拟合三个AFT模型（如果相应文件可用）
if(!is.null(mid_file)) fit_mid <- fit_survreg(mid_file, "mid")
if(!is.null(low_file)) fit_low <- fit_survreg(low_file, "low")
if(!is.null(up_file)) fit_up <- fit_survreg(up_file, "up")

# -------- 拟合IC-AFT模型 --------
cat("Fitting ic_par (IC-AFT) ...\n")
ic_fit <- tryCatch({
  if(!"icenReg" %in% (.packages())) {
    cat("Loading icenReg package\n")
    library(icenReg)
  }
  icenReg::ic_par(Surv(GA_lower, GA_upper, type = "interval2") ~ BMI, 
                 data = intervals, dist = "lognormal")
}, error = function(e) {
  cat("ic_par(lognormal) failed with error:\n")
  print(e)
  cat("Trying ic_par with weibull ...\n")
  tryCatch(
    icenReg::ic_par(Surv(GA_lower, GA_upper, type = "interval2") ~ BMI, 
                   data = intervals, dist = "weibull"),
    error = function(e2) {
      cat("ic_par fallback also failed. Will output empty icenreg_out.csv\n")
      NULL
    }
  )
})

# -------- 计算不同BMI分位数下的生存率 --------
if(!is.null(ic_fit)) {
  # 设置BMI分位点
  bmi_quantiles <- quantile(intervals$BMI, probs = c(0.25, 0.5, 0.75), na.rm = TRUE)
  # 将BMI分位数写入文件以供MATLAB使用
  write.table(bmi_quantiles, file.path(out_dir, "icenreg_bmi_levels.csv"), 
              row.names = FALSE, col.names = FALSE)
  
  # 网格点
  tgrid <- seq(10, 25, by = 0.1)
  
  # 计算各BMI分位数的生存率
  S_mat <- matrix(NA, nrow = length(tgrid), ncol = length(bmi_quantiles))
  for(i in 1:length(bmi_quantiles)) {
    newdata <- data.frame(BMI = bmi_quantiles[i])
    tryCatch({
      S_mat[,i] <- predict(ic_fit, newdata = newdata, times = tgrid, type = "survival")
    }, error = function(e) {
      cat("Error predicting survival for BMI =", bmi_quantiles[i], ":", conditionMessage(e), "\n")
      S_mat[,i] <- rep(1, length(tgrid))  # 保守估计
    })
  }
  
  # 生成输出DataFrame并保存
  out_df <- data.frame(week = tgrid)
  for(i in 1:length(bmi_quantiles)) {
    out_df[paste0("S_BMIq", i)] <- S_mat[,i]
  }
  
  # 根据不同情况输出到不同文件
  if(bootstrap_flag) {
    # 从文件名提取ID
    batch_id <- sub(".*_([0-9]+)\\.csv$", "\\1", intervals_file)
    if(batch_id != intervals_file) {
      out_file <- file.path(out_dir, paste0("icenreg_boot_out_", batch_id, ".csv"))
      write.table(cbind(tgrid, S_mat), out_file, row.names = FALSE, col.names = FALSE, sep = ",")
      cat("Wrote bootstrap output to", out_file, "\n")
    } else {
      # MC样本
      batch_id <- sub(".*batch_([0-9]+)\\.csv$", "\\1", intervals_file)
      if(batch_id != intervals_file) {
        out_file <- file.path(out_dir, paste0("icenreg_mc_out_", batch_id, ".csv"))
        write.table(cbind(tgrid, S_mat), out_file, row.names = FALSE, col.names = FALSE, sep = ",")
        cat("Wrote Monte Carlo output to", out_file, "\n")
      } else {
        # 标准输出
        write.csv(out_df, file.path(out_dir, "icenreg_out.csv"), row.names = FALSE)
        cat("Wrote standard output to", file.path(out_dir, "icenreg_out.csv"), "\n")
      }
    }
  } else {
    # 标准输出
    write.csv(out_df, file.path(out_dir, "icenreg_out.csv"), row.names = FALSE)
    cat("Wrote standard output to", file.path(out_dir, "icenreg_out.csv"), "\n")
  }
} else {
  # 创建空的输出文件，防止MATLAB报错
  if(bootstrap_flag) {
    batch_id <- sub(".*_([0-9]+)\\.csv$", "\\1", intervals_file)
    if(batch_id != intervals_file) {
      empty_file <- file.path(out_dir, paste0("icenreg_boot_out_", batch_id, ".csv"))
      write.table(matrix(c(seq(10, 25, 0.1), rep(1, length(seq(10, 25, 0.1)))), ncol=2), 
                  empty_file, row.names=FALSE, col.names=FALSE)
      cat("Created empty bootstrap output", empty_file, "\n")
    } else {
      batch_id <- sub(".*batch_([0-9]+)\\.csv$", "\\1", intervals_file)
      if(batch_id != intervals_file) {
        empty_file <- file.path(out_dir, paste0("icenreg_mc_out_", batch_id, ".csv"))
        write.table(matrix(c(seq(10, 25, 0.1), rep(1, length(seq(10, 25, 0.1)))), ncol=2), 
                    empty_file, row.names=FALSE, col.names=FALSE)
        cat("Created empty MC output", empty_file, "\n")
      }
    }
  }
  empty_main <- file.path(out_dir, "icenreg_out.csv")
  out_df <- data.frame(
    week = seq(10, 25, 0.1),
    S_BMIq1 = 1,
    S_BMIq2 = 1,
    S_BMIq3 = 1
  )
  write.csv(out_df, empty_main, row.names = FALSE)
  cat("Created empty main output file", empty_main, "\n")
}

cat("icenreg.R completed\n")