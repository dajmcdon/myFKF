% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/ssmodelfuncs.R
\name{namedTrends}
\alias{namedTrends}
\title{Specific trend models}
\usage{
namedTrends(trendType, parmlist)
}
\arguments{
\item{trendType}{The name of a specific trend type. Options are
\code{LocalLevel}, \code{LocalLinTrend}, \code{SmoothSlope},
\code{ConstantSlope}, and \code{IntRandWalk}. Input should be a quoted
string. This updates the parmlist to reflect the specific trend choice. Any
other string simply applies \code{\link{trendModelArb}} to the parmlist.}

\item{parmlist}{A list of parameters as generated by \code{\link{parmList}}.}
}
\value{
A list of matrices with components \code{Z}, \code{Tt}, \code{Q}, and
  \code{R}.
}
\description{
This is simply a wrapper for \code{\link{trendModelArb}} which produces
specific trend models discussed in Chapter 3 of Durbin and Koopman.
}
\details{
The named trend models have the following forms:

1. LocalLevel: \deqn{y_t = mu_t + more} \deqn{mu_t+1 = mu_t + xi_t} 2.
LocalLinTrend: \deqn{y_t = mu_t + more} \deqn{mu_t+1 = mu_t + nu_t + xi_t}
\deqn{nu_t+1 = nu_t + eta_t} 3. SmoothSlope: \deqn{y_t = mu_t + more}
\deqn{mu_t+1 = mu_t + nu_t} \deqn{nu_t+1 = nu_t + eta_t} 4. ConstantSlope has
trend = constSlope, gives linear trend \deqn{y_t = mu_t + more} \deqn{mu_t+1
= constSlope + mu_t} 5. IRW(r), r > 1,default is r=2 \deqn{y_t = mu_t + more}
\deqn{mu_t+1 = 2mu_t - mu_t-1 + xi_t}
}
\references{
Durbin, James, and Koopman, Siem Jan. \emph{Time Series Analysis
by State Space Methods} (2nd Edition). Cambridge, GBR: OUP Oxford, 2012.
}
