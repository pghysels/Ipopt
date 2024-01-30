// Copyright (C) 2004, 2010 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// Authors:  Carl Laird, Andreas Waechter     IBM    2005-03-17

#include "IpoptConfig.h"
#include "IpMa27TSolverInterface.hpp"

#include <cmath>
#include <sstream>
#include <fstream>
#include <numeric>
#include <iomanip>

#if defined(QD_IR)
#include <qd/qd_real.h>
#endif

#ifdef IPOPT_HAS_HSL
#include "CoinHslConfig.h"
#endif

#if (defined(COINHSL_HAS_MA27) && !defined(IPOPT_SINGLE)) || (defined(COINHSL_HAS_MA27S) && defined(IPOPT_SINGLE))
#ifdef IPOPT_SINGLE
#define IPOPT_HSL_FUNCP(name,NAME) IPOPT_HSL_FUNC(name,NAME)
#else
#define IPOPT_HSL_FUNCP(name,NAME) IPOPT_HSL_FUNC(name ## d,NAME ## D)
#endif

/** MA27 functions from HSL library (symbols resolved at linktime) */
extern "C"
{
   IPOPT_DECL_MA27A(IPOPT_HSL_FUNCP(ma27a, MA27A));
   IPOPT_DECL_MA27B(IPOPT_HSL_FUNCP(ma27b, MA27B));
   IPOPT_DECL_MA27C(IPOPT_HSL_FUNCP(ma27c, MA27C));
   IPOPT_DECL_MA27I(IPOPT_HSL_FUNCP(ma27i, MA27I));
}
#else
#ifdef IPOPT_SINGLE
#define HSLFUNCNAMESUFFIX ""
#else
#define HSLFUNCNAMESUFFIX "d"
#endif
#endif

namespace Ipopt
{
#if IPOPT_VERBOSITY > 0
static const Index dbg_verbosity = 0;
#endif

/** pointer to MA27 function that can be set via Ma27TSolverInterface::SetFunctions() */
static IPOPT_DECL_MA27A(*user_ma27a) = NULL;
static IPOPT_DECL_MA27B(*user_ma27b) = NULL;
static IPOPT_DECL_MA27C(*user_ma27c) = NULL;
static IPOPT_DECL_MA27I(*user_ma27i) = NULL;

Ma27TSolverInterface::Ma27TSolverInterface(
   SmartPtr<LibraryLoader> hslloader_
)  : hslloader(hslloader_),
   ma27a(NULL),
   ma27b(NULL),
   ma27c(NULL),
   ma27i(NULL),
   dim_(0),
   nonzeros_(0),
   initialized_(false),
   pivtol_changed_(false),
   refactorize_(false),
   liw_(0),
   iw_(NULL),
   ikeep_(NULL),
   la_(0),
   a_(NULL),
   la_increase_(false),
   liw_increase_(false)
{
   DBG_START_METH("Ma27TSolverInterface::Ma27TSolverInterface()", dbg_verbosity);
}

Ma27TSolverInterface::~Ma27TSolverInterface()
{
   DBG_START_METH("Ma27TSolverInterface::~Ma27TSolverInterface()",
                  dbg_verbosity);
   delete[] iw_;
   delete[] ikeep_;
   delete[] a_;

#if defined(QD_IR)
   delete[] airn_orig_;
   delete[] ajcn_orig_;
   delete[] a_orig_;
#endif

}

void Ma27TSolverInterface::RegisterOptions(
   SmartPtr<RegisteredOptions> roptions
)
{
   roptions->AddBoundedIntegerOption(
      "ma27_print_level",
      "Debug printing level for the linear solver MA27",
      0, 4, 0,
      "0: no printing; 1: Error messages only; 2: Error and warning messages; 3: Error and warning messages and terse monitoring; 4: All information.");
   roptions->AddBoundedNumberOption(
      "ma27_pivtol",
      "Pivot tolerance for the linear solver MA27.",
      0.0, true,
      1.0, true,
      1e-8,
      "A smaller number pivots for sparsity, a larger number pivots for stability.");
   roptions->AddBoundedNumberOption(
      "ma27_pivtolmax",
      "Maximum pivot tolerance for the linear solver MA27.",
      0.0, true,
      1.0, true,
      1e-4,
      "Ipopt may increase pivtol as high as ma27_pivtolmax to get a more accurate solution to the linear system.");
   roptions->AddLowerBoundedNumberOption(
      "ma27_liw_init_factor",
      "Integer workspace memory for MA27.",
      1.0, false,
      5.0,
      "The initial integer workspace memory = liw_init_factor * memory required by unfactored system. "
      "Ipopt will increase the workspace size by ma27_meminc_factor if required.");
   roptions->AddLowerBoundedNumberOption(
      "ma27_la_init_factor",
      "Real workspace memory for MA27.",
      1.0, false,
      5.0,
      "The initial real workspace memory = la_init_factor * memory required by unfactored system. "
      "Ipopt will increase the workspace size by ma27_meminc_factor if required.");
   roptions->AddLowerBoundedNumberOption(
      "ma27_meminc_factor",
      "Increment factor for workspace size for MA27.",
      1.0, false,
      2.0,
      "If the integer or real workspace is not large enough, Ipopt will increase its size by this factor.");
   roptions->AddBoolOption(
      "ma27_skip_inertia_check",
      "Whether to always pretend that inertia is correct.",
      false,
      "Setting this option to \"yes\" essentially disables inertia check. "
      "This option makes the algorithm non-robust and easily fail, but it might give some insight into the necessity of inertia control.",
      true);
   roptions->AddBoolOption(
      "ma27_ignore_singularity",
      "Whether to use MA27's ability to solve a linear system even if the matrix is singular.",
      false,
      "Setting this option to \"yes\" means that Ipopt will call MA27 to compute solutions for right hand sides, "
      "even if MA27 has detected that the matrix is singular (but is still able to solve the linear system). "
      "In some cases this might be better than using Ipopt's heuristic of small perturbation of the lower diagonal of the KKT matrix.",
      true);
}

void Ma27TSolverInterface::SetFunctions(
   IPOPT_DECL_MA27A(*ma27a),
   IPOPT_DECL_MA27B(*ma27b),
   IPOPT_DECL_MA27C(*ma27c),
   IPOPT_DECL_MA27I(*ma27i)
)
{
   DBG_ASSERT(ma27a != NULL);
   DBG_ASSERT(ma27b != NULL);
   DBG_ASSERT(ma27c != NULL);
   DBG_ASSERT(ma27i != NULL);

   user_ma27a = ma27a;
   user_ma27b = ma27b;
   user_ma27c = ma27c;
   user_ma27i = ma27i;
}

bool Ma27TSolverInterface::InitializeImpl(
   const OptionsList& options,
   const std::string& prefix
)
{
   if( user_ma27a != NULL )
   {
      // someone set MA27 functions via setFunctions - prefer these
      ma27a = user_ma27a;
      ma27b = user_ma27b;
      ma27c = user_ma27c;
      ma27i = user_ma27i;
   }
   else
   {
#if (defined(COINHSL_HAS_MA27) && !defined(IPOPT_SINGLE)) || (defined(COINHSL_HAS_MA27S) && defined(IPOPT_SINGLE))
      // use HSL functions that should be available in linked HSL library
      ma27a = &::IPOPT_HSL_FUNCP(ma27a, MA27A);
      ma27b = &::IPOPT_HSL_FUNCP(ma27b, MA27B);
      ma27c = &::IPOPT_HSL_FUNCP(ma27c, MA27C);
      ma27i = &::IPOPT_HSL_FUNCP(ma27i, MA27I);
#else
      DBG_ASSERT(IsValid(hslloader));

      ma27a = (IPOPT_DECL_MA27A(*))hslloader->loadSymbol("ma27a" HSLFUNCNAMESUFFIX);
      ma27b = (IPOPT_DECL_MA27B(*))hslloader->loadSymbol("ma27b" HSLFUNCNAMESUFFIX);
      ma27c = (IPOPT_DECL_MA27C(*))hslloader->loadSymbol("ma27c" HSLFUNCNAMESUFFIX);
      ma27i = (IPOPT_DECL_MA27I(*))hslloader->loadSymbol("ma27i" HSLFUNCNAMESUFFIX);
#endif
   }

   DBG_ASSERT(ma27a != NULL);
   DBG_ASSERT(ma27b != NULL);
   DBG_ASSERT(ma27c != NULL);
   DBG_ASSERT(ma27i != NULL);

   options.GetNumericValue("ma27_pivtol", pivtol_, prefix);
   if( options.GetNumericValue("ma27_pivtolmax", pivtolmax_, prefix) )
   {
      ASSERT_EXCEPTION(pivtolmax_ >= pivtol_, OPTION_INVALID, "Option \"ma27_pivtolmax\": This value must be between "
                       "ma27_pivtol and 1.");
   }
   else
   {
      pivtolmax_ = Max(pivtolmax_, pivtol_);
   }

   Index print_level;
   options.GetIntegerValue("ma27_print_level", print_level, prefix);
   options.GetNumericValue("ma27_liw_init_factor", liw_init_factor_, prefix);
   options.GetNumericValue("ma27_la_init_factor", la_init_factor_, prefix);
   options.GetNumericValue("ma27_meminc_factor", meminc_factor_, prefix);
   options.GetBoolValue("ma27_skip_inertia_check", skip_inertia_check_, prefix);
   options.GetBoolValue("ma27_ignore_singularity", ignore_singularity_, prefix);
   // The following option is registered by OrigIpoptNLP
   options.GetBoolValue("warm_start_same_structure", warm_start_same_structure_, prefix);

   /* Set the default options for MA27 */
   ma27i(icntl_, cntl_);

   if( print_level == 0 )
   {
      icntl_[0] = 0;   // Suppress error messages
   }
   if( print_level <= 1 )
   {
      icntl_[1] = 0;   // Suppress warning messages
   }
   if( print_level >= 2 )
   {
      icntl_[2] = print_level - 2;   // diagnostic messages level
   }

   // Reset all private data
   initialized_ = false;
   pivtol_changed_ = false;
   refactorize_ = false;

   la_increase_ = false;
   liw_increase_ = false;

   if( !warm_start_same_structure_ )
   {
      dim_ = 0;
      nonzeros_ = 0;
   }
   else
   {
      ASSERT_EXCEPTION(dim_ > 0 && nonzeros_ > 0, INVALID_WARMSTART,
                       "Ma27TSolverInterface called with warm_start_same_structure, but the problem is solved for the first time.");
   }

   return true;
}

ESymSolverStatus Ma27TSolverInterface::MultiSolve(
   bool         new_matrix,
   const Index* airn,
   const Index* ajcn,
   Index        nrhs,
   Number*      rhs_vals,
   bool         check_NegEVals,
   Index        numberOfNegEVals
)
{
   DBG_START_METH("Ma27TSolverInterface::MultiSolve", dbg_verbosity);
   DBG_ASSERT(!check_NegEVals || ProvidesInertia());
   DBG_ASSERT(initialized_);
   DBG_ASSERT(la_ != 0);

   if( pivtol_changed_ )
   {
      DBG_PRINT((1, "Pivot tolerance has changed.\n"));
      pivtol_changed_ = false;
      // If the pivot tolerance has been changed but the matrix is not
      // new, we have to request the values for the matrix again to do
      // the factorization again.
      if( !new_matrix )
      {
         DBG_PRINT((1, "Ask caller to call again.\n"));
         refactorize_ = true;
         return SYMSOLVER_CALL_AGAIN;
      }
   }

   // check if a factorization has to be done
   DBG_PRINT((1, "new_matrix = %d\n", new_matrix));
   if( new_matrix || refactorize_ )
   {
      // perform the factorization
      ESymSolverStatus retval;
      retval = Factorization(airn, ajcn, check_NegEVals, numberOfNegEVals);
      if( retval != SYMSOLVER_SUCCESS )
      {
         DBG_PRINT((1, "FACTORIZATION FAILED!\n"));
         return retval;  // Matrix singular or error occurred
      }
      refactorize_ = false;
   }

   // do the backsolve
   return Backsolve(nrhs, rhs_vals);
}

Number* Ma27TSolverInterface::GetValuesArrayPtr()
{
   DBG_START_METH("Ma27TSolverInterface::GetValuesArrayPtr", dbg_verbosity);
   DBG_ASSERT(initialized_);

   // If the size of a is to be increase for the next factorization
   // anyway, delete the current large array and just return enough
   // to store the values

   if( la_increase_ )
   {
      delete[] a_;
      a_ = NULL;
      a_ = new Number[nonzeros_];
   }

   return a_;
}

/** Initialize the local copy of the positions of the nonzero elements */
ESymSolverStatus Ma27TSolverInterface::InitializeStructure(
   Index        dim,
   Index        nonzeros,
   const Index* airn,
   const Index* ajcn
)
{
   DBG_START_METH("Ma27TSolverInterface::InitializeStructure", dbg_verbosity);

   ESymSolverStatus retval = SYMSOLVER_SUCCESS;
   if( !warm_start_same_structure_ )
   {
      dim_ = dim;
      nonzeros_ = nonzeros;

      // Do the symbolic facotrization
      retval = SymbolicFactorization(airn, ajcn);
      if( retval != SYMSOLVER_SUCCESS )
      {
         return retval;
      }
   }
   else
   {
      ASSERT_EXCEPTION(dim_ == dim && nonzeros_ == nonzeros, INVALID_WARMSTART,
                       "Ma27TSolverInterface called with warm_start_same_structure, but the problem size has changed.");
   }

   initialized_ = true;

   return retval;
}

ESymSolverStatus Ma27TSolverInterface::SymbolicFactorization(
   const Index* airn,
   const Index* ajcn
)
{
   DBG_START_METH("Ma27TSolverInterface::SymbolicFactorization", dbg_verbosity);

   if( HaveIpData() )
   {
      IpData().TimingStats().LinearSystemSymbolicFactorization().Start();
   }

   // Get memory for the IW workspace
   delete[] iw_;
   iw_ = NULL;

   // Overestimation factor for LIW (20% recommended in MA27 documentation)
   const Number LiwFact = 2.0;      // This is 100% overestimation
   Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA,
                  "In Ma27TSolverInterface::InitializeStructure: Using overestimation factor LiwFact = %e\n", LiwFact);
   liw_ = (Index) (LiwFact * (Number(2 * nonzeros_ + 3 * dim_ + 1)));
   try
   {
      iw_ = new Index[liw_];
   }
   catch( const std::bad_alloc& )
   {
      Jnlst().Printf(J_STRONGWARNING, J_LINEAR_ALGEBRA, "Failed to allocate initial working space (iw_) for MA27\n");
      throw; // will be caught in IpIpoptApplication
   }

   // Get memory for IKEEP
   delete[] ikeep_;
   ikeep_ = NULL;
   ikeep_ = new Index[3 * dim_];

   if( Jnlst().ProduceOutput(J_MOREMATRIX, J_LINEAR_ALGEBRA) )
   {
      Jnlst().Printf(J_MOREMATRIX, J_LINEAR_ALGEBRA,
                     "\nMatrix structure given to MA27 with dimension %" IPOPT_INDEX_FORMAT " and %" IPOPT_INDEX_FORMAT " nonzero entries:\n", dim_, nonzeros_);
      for( Index i = 0; i < nonzeros_; i++ )
      {
         Jnlst().Printf(J_MOREMATRIX, J_LINEAR_ALGEBRA,
                        "A[%5d,%5d]\n", airn[i], ajcn[i]);
      }
   }

   // Call MA27AX
   Index N = dim_;
   Index NZ = nonzeros_;
   Index IFLAG = 0;
   Number OPS;
   Index INFO[20];
   Index* IW1 = new Index[2 * dim_];      // Get memory for IW1 (only local)
   ma27a(&N, &NZ, airn, ajcn, iw_, &liw_, ikeep_, IW1, &nsteps_, &IFLAG, icntl_, cntl_, INFO, &OPS);
   delete[] IW1;      // No longer required

   // Receive several information
   const Index& iflag = INFO[0];      // Information flag
   const Index& ierror = INFO[1];      // Error flag
   const Index& nrlnec = INFO[4];      // recommended value for la
   const Index& nirnec = INFO[5];      // recommended value for liw

   Jnlst().Printf(J_MOREDETAILED, J_LINEAR_ALGEBRA,
                  "Return values from MA27AD: IFLAG = %" IPOPT_INDEX_FORMAT ", IERROR = %" IPOPT_INDEX_FORMAT "\n", iflag, ierror);

   // Check if error occurred
   if( iflag != 0 )
   {
      Jnlst().Printf(J_ERROR, J_LINEAR_ALGEBRA,
                     "*** Error from MA27AD *** IFLAG = %" IPOPT_INDEX_FORMAT " IERROR = %" IPOPT_INDEX_FORMAT "\n", iflag, ierror);
      if( iflag == 1 )
         Jnlst().Printf(J_ERROR, J_LINEAR_ALGEBRA,
                        "The index of a matrix is out of range.\nPlease check your implementation of the Jacobian and Hessian matrices.\n");
      if( HaveIpData() )
      {
         IpData().TimingStats().LinearSystemSymbolicFactorization().End();
      }
      return SYMSOLVER_FATAL_ERROR;
   }

   try
   {
      // Reserve memory for iw_ for later calls, based on suggested size
      delete[] iw_;
      iw_ = NULL;
      Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA,
                     "Size of integer work space recommended by MA27 is %" IPOPT_INDEX_FORMAT "\n", nirnec);
      ComputeMemIncrease(liw_, liw_init_factor_ * (Number) nirnec, 0, "integer working space for MA27");
      Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA,
                     "Setting integer work space size to %" IPOPT_INDEX_FORMAT "\n", liw_);
      iw_ = new Index[liw_];

      // Reserve memory for a_
      delete[] a_;
      a_ = NULL;
      Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA,
                     "Size of doublespace recommended by MA27 is %" IPOPT_INDEX_FORMAT "\n", nrlnec);
      ComputeMemIncrease(la_, la_init_factor_ * (Number) nrlnec, nonzeros_, "double working space for MA27");
      Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA,
                     "Setting double work space size to %" IPOPT_INDEX_FORMAT "\n", la_);
      a_ = new Number[la_];
   }
   catch( const std::bad_alloc& )
   {
      Jnlst().Printf(J_STRONGWARNING, J_LINEAR_ALGEBRA, "Failed to allocate more working space for MA27\n");
      throw; // will be caught in IpIpoptApplication
   }

   if( HaveIpData() )
   {
      IpData().TimingStats().LinearSystemSymbolicFactorization().End();
   }

   return SYMSOLVER_SUCCESS;
}

ESymSolverStatus Ma27TSolverInterface::Factorization(
   const Index* airn,
   const Index* ajcn,
   bool         check_NegEVals,
   Index        numberOfNegEVals
)
{
   DBG_START_METH("Ma27TSolverInterface::Factorization", dbg_verbosity);

   // Check if la should be increased
   if( HaveIpData() )
   {
      IpData().TimingStats().LinearSystemFactorization().Start();
   }

   if( la_increase_ )
   {
      Number* a_old = a_;
      Index la_old = la_;
      ComputeMemIncrease(la_, meminc_factor_ * (Number) la_, 0, "double working space for MA27");
      Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA,
                     "In Ma27TSolverInterface::Factorization: Increasing la from %" IPOPT_INDEX_FORMAT " to %" IPOPT_INDEX_FORMAT "\n", la_old, la_);
      try
      {
         a_ = new Number[la_];
      }
      catch( const std::bad_alloc& )
      {
         Jnlst().Printf(J_STRONGWARNING, J_LINEAR_ALGEBRA, "Failed to allocate more working space (a_) for MA27\n");
         throw; // will be caught in IpIpoptApplication
      }
      for( Index i = 0; i < nonzeros_; i++ )
      {
         a_[i] = a_old[i];
      }
      delete[] a_old;
      la_increase_ = false;
   }

   // Check if liw should be increased
   if( liw_increase_ )
   {
      delete[] iw_;
      iw_ = NULL;
      Index liw_old = liw_;
      ComputeMemIncrease(liw_, meminc_factor_ * (Number) liw_, 0, "integer working space for MA27");
      Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA,
                     "In Ma27TSolverInterface::Factorization: Increasing liw from %" IPOPT_INDEX_FORMAT " to %" IPOPT_INDEX_FORMAT "\n", liw_old, liw_);
      try
      {
         iw_ = new Index[liw_];
      }
      catch( const std::bad_alloc& )
      {
         Jnlst().Printf(J_STRONGWARNING, J_LINEAR_ALGEBRA, "Failed to allocate more working space (iw_) for MA27\n");
         throw; // will be caught in IpIpoptApplication
      }
      liw_increase_ = false;
   }

   Index iflag;  // Information flag
   Index ncmpbr;  // Number of double precision compressions
   Index ncmpbi;  // Number of integer compressions

   // Call MA27BX; possibly repeatedly if workspaces are too small
   Index N = dim_;
   Index NZ = nonzeros_;
   Index* IW1 = new Index[2 * dim_];
   Index INFO[20];
   cntl_[0] = pivtol_;  // Set pivot tolerance

#if defined(QD_IR)
   NZ_orig_ = NZ;
   airn_orig_ = new Index[NZ];
   ajcn_orig_ = new Index[NZ];
   a_orig_ = new Number[NZ];
   std::copy(airn, airn+NZ, airn_orig_);
   std::copy(ajcn, ajcn+NZ, ajcn_orig_);
   std::copy(a_, a_+NZ, a_orig_);
#endif
   //////////////////////////////////////////////////////////////////////


   ma27b(&N, &NZ, airn, ajcn, a_, &la_, iw_, &liw_, ikeep_, &nsteps_, &maxfrt_, IW1, icntl_, cntl_, INFO);
   delete[] IW1;

   // Receive information about the factorization
   iflag = INFO[0];  // Information flag
   const Index& ierror = INFO[1];  // Error flag
   ncmpbr = INFO[11];  // Number of double compressions
   ncmpbi = INFO[12];  // Number of integer compressions
   negevals_ = INFO[14];  // Number of negative eigenvalues

   Jnlst().Printf(J_MOREDETAILED, J_LINEAR_ALGEBRA,
                  "Return values from MA27BD: IFLAG = %" IPOPT_INDEX_FORMAT ", IERROR = %" IPOPT_INDEX_FORMAT "\n", iflag, ierror);

   DBG_PRINT((1, "Return from MA27BD iflag = %" IPOPT_INDEX_FORMAT " and ierror = %" IPOPT_INDEX_FORMAT "\n",
              iflag, ierror));

   // Check if factorization failed due to insufficient memory space
   // iflag==-3 if LIW too small (recommended value in ierror)
   // iflag==-4 if LA too small (recommended value in ierror)
   if( iflag == -3 || iflag == -4 )
   {
      // Increase size of both LIW and LA
      delete[] iw_;
      iw_ = NULL;
      delete[] a_;
      a_ = NULL;
      Index liw_old = liw_;
      Index la_old = la_;
      if( iflag == -3 )
      {
         ComputeMemIncrease(liw_, meminc_factor_ * (Number) ierror, 0, "integer working space for MA27");
         ComputeMemIncrease(la_, meminc_factor_ * (Number) la_, 0, "double working space for MA27");
      }
      else
      {
         ComputeMemIncrease(liw_, meminc_factor_ * (Number) liw_, 0, "integer working space for MA27");
         ComputeMemIncrease(la_, meminc_factor_ * (Number) ierror, 0, "double working space for MA27");
      }
      Jnlst().Printf(J_WARNING, J_LINEAR_ALGEBRA,
                     "MA27BD returned iflag=%" IPOPT_INDEX_FORMAT " and requires more memory.\n Increase liw from %" IPOPT_INDEX_FORMAT " to %" IPOPT_INDEX_FORMAT " and la from %" IPOPT_INDEX_FORMAT " to %" IPOPT_INDEX_FORMAT " and factorize again.\n",
                     iflag, liw_old, liw_, la_old, la_);
      try
      {
         iw_ = new Index[liw_];
         a_ = new Number[la_];
      }
      catch( const std::bad_alloc& )
      {
         Jnlst().Printf(J_STRONGWARNING, J_LINEAR_ALGEBRA, "Failed to allocate more working space (iw_ and a_) for MA27\n");
         throw; // will be caught in IpIpoptApplication
      }
      if( HaveIpData() )
      {
         IpData().TimingStats().LinearSystemFactorization().End();
      }
      return SYMSOLVER_CALL_AGAIN;
   }

   // Check if the system is singular, and if some other error occurred
   if( iflag == -5 || (!ignore_singularity_ && iflag == 3) )
   {
      if( HaveIpData() )
      {
         IpData().TimingStats().LinearSystemFactorization().End();
      }
      return SYMSOLVER_SINGULAR;
   }
   else if( iflag == 3 )
   {
      Index missing_rank = dim_ - INFO[1];
      Jnlst().Printf(J_WARNING, J_LINEAR_ALGEBRA,
                     "MA27BD returned iflag=%" IPOPT_INDEX_FORMAT " and detected rank deficiency of degree %" IPOPT_INDEX_FORMAT ".\n", iflag, missing_rank);
      // We correct the number of negative eigenvalues here to include
      // the zero eigenvalues, since otherwise we indicate the wrong
      // inertia.
      negevals_ += missing_rank;
   }
   else if( iflag != 0 )
   {
      // There is some error
      if( HaveIpData() )
      {
         IpData().TimingStats().LinearSystemFactorization().End();
      }
      return SYMSOLVER_FATAL_ERROR;
   }

   // Check if it might be more efficient to use more memory next time
   // (if there were too many compressions for this factorization)
   if( ncmpbr >= 10 )
   {
      la_increase_ = true;
      Jnlst().Printf(J_WARNING, J_LINEAR_ALGEBRA,
                     "MA27BD returned ncmpbr=%" IPOPT_INDEX_FORMAT ". Increase la before the next factorization.\n", ncmpbr);
   }
   if( ncmpbi >= 10 )
   {
      liw_increase_ = true;
      Jnlst().Printf(J_WARNING, J_LINEAR_ALGEBRA,
                     "MA27BD returned ncmpbi=%" IPOPT_INDEX_FORMAT ". Increase liw before the next factorization.\n", ncmpbr);
   }

   Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA,
                  "Number of doubles for MA27 to hold factorization (INFO(9)) = %" IPOPT_INDEX_FORMAT "\n", INFO[8]);
   Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA,
                  "Number of integers for MA27 to hold factorization (INFO(10)) = %" IPOPT_INDEX_FORMAT "\n", INFO[9]);

   // Check whether the number of negative eigenvalues matches the requested
   // count
   if( HaveIpData() )
   {
      IpData().TimingStats().LinearSystemFactorization().End();
   }
   if( !skip_inertia_check_ && check_NegEVals && (numberOfNegEVals != negevals_) )
   {
      Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA,
                     "In Ma27TSolverInterface::Factorization: negevals_ = %" IPOPT_INDEX_FORMAT ", but numberOfNegEVals = %" IPOPT_INDEX_FORMAT "\n", negevals_,
                     numberOfNegEVals);
      return SYMSOLVER_WRONG_INERTIA;
   }

   return SYMSOLVER_SUCCESS;
}

#if defined(QD_IR)
template<typename it> qd_real qd_norm(const it v0, const it v1) {
  qd_real nrm(0.);
  for (auto v=v0; v!=v1; v++)
    nrm += qd_real(*v)*qd_real(*v);
  return sqrt(nrm);
};

template<typename T> qd_real qd_norm(const std::vector<T>& v) {
  return qd_norm(v.cbegin(), v.cend());
};

qd_real qd_dot(const std::vector<qd_real>& a,
               const std::vector<qd_real>& b) {
  return std::inner_product
    (a.cbegin(), a.cend(), b.cbegin(), qd_real(0));
}

template<typename T>
void qd_axpy(qd_real alpha, const std::vector<T>& x,
             std::vector<qd_real>& y) {
  for (std::size_t i=0; i<x.size(); i++)
    y[i] += alpha * qd_real(x[i]);
}

void to_Number(const std::vector<qd_real>& a, Number* b) {
  // TODO what if Number != double ?
  for (std::size_t i=0; i<a.size(); i++)
    b[i] = to_double(a[i]);
}

template<typename SPMV>
void qd_residual(const SPMV& spmv, const std::vector<qd_real>& x,
                 const Number* b, std::vector<qd_real>& r) {
  spmv(x, r);
  for (std::size_t i=0; i<x.size(); i++)
    r[i] = qd_real(b[i]) - r[i];
}

template<typename SPMV, typename PREC>
void qd_refinement(Index N, const SPMV& spmv, const PREC& prec, Number* b) {
  qd_real tol("1e-60"), r_nrm_old(0), b_nrm = qd_norm(b, b+N);
  std::vector<qd_real> r(N), x(N, qd_real(0.)), x_old(N);
  r.assign(b, b+N);
  int maxit = 20;
  for (int ref=0; ref<maxit; ref++) {
    prec(r);
    if (ref > 1) x_old = x;
    qd_axpy(qd_real(1.), r, x);
    auto e_nrm = qd_norm(r);
    qd_residual(spmv, x, b, r);
    auto r_nrm = qd_norm(r);
    auto r_nrm_rel_b = r_nrm / b_nrm;
    auto print_IR = [&]() {
      std::cout << " it " << ref << " ||e||= " << e_nrm
                << " ||r||= " << r_nrm
                << " ||r||/||b||= " << r_nrm_rel_b << std::endl;
    };
    print_IR();
    if (ref > 0 && r_nrm > r_nrm_old) {
      x = x_old;
      r_nrm_old = r_nrm;
      std::cout << "QDIR divergence!";
      print_IR();
      break;
    }
    if (e_nrm <= tol || r_nrm_rel_b <= tol) {
      print_IR();
      break;
    }
    if (ref == maxit-1) {
      std::cout << "QDIR max-iterations!";
      print_IR();
      break;
    }
    r_nrm_old = r_nrm;
  }
  to_Number(x, b);
}

/*
 * right preconditioned flexible GMRES
 * https://www-users.cse.umn.edu/~saad/PDF/umsi-91-279.pdf
 *
 * We want this to be flexible because the rounding is not linear?
 */
template<typename SPMV, typename PREC>
void qd_gmres(Index N, const SPMV& spmv, const PREC& prec, Number* b) {
  int m = 10, maxit = 100, totit = 0, ldh = m + 1;
  if (m > maxit) m = maxit;
  bool no_conv = true;
  qd_real tol("1e-60"), rho0(0.);
  std::vector<qd_real> x(N);
  while (no_conv) {
    std::vector<qd_real> givens_c(m), givens_s(m),
      b_(m+1), hess(m*(m+1));
    std::vector<std::vector<qd_real>> V, Z;
    V.emplace_back(N);
    qd_residual(spmv, x, b, V[0]);
    auto rho = qd_norm(V[0]);
    if (totit == 0) rho0 = rho;
    if (rho/rho0 < tol || rho < tol) { no_conv = false; break; }
    for (Index i=0; i<N; i++)
      V[0][i] /= rho;
    b_[0] = rho;
    for (int i=1; i<=m; i++)
      b_[i] = qd_real(0.);
    int nrit = m - 1;
    // std::cout << "GMRES it. " << totit << "\tres = "
    //           << std::setw(12) << rho
    //           << "\trel.res = " << std::setw(12)
    //           << rho/rho0 << "\t restart!" << std::endl;
    // for (int it=0; it<m; it++) {
    for (int it=0; it<m; it++) {
      totit++;
      Z.emplace_back(V[it]);
      prec(Z[it]);
      V.emplace_back(N);
      spmv(Z[it], V[it+1]);
      for (int k=0; k<=it; k++) {
        hess[k+it*ldh] = qd_dot(V[k], V[it+1]);
        qd_axpy(-hess[k+it*ldh], V[k], V[it+1]);
      }
      hess[it+1+it*ldh] = qd_norm(V[it+1]);
      for (Index i=0; i<N; i++)
        V[it+1][i] /= hess[it+1+it*ldh];
      for (int k=1; k<it+1; k++) {
        qd_real gamma = givens_c[k-1]*hess[k-1+it*ldh]
          + givens_s[k-1]*hess[k+it*ldh];
        hess[k+it*ldh] = -givens_s[k-1]*hess[k-1+it*ldh]
          + givens_c[k-1]*hess[k+it*ldh];
        hess[k-1+it*ldh] = gamma;
      }
      qd_real delta =
        sqrt(hess[it+it*ldh]*hess[it+it*ldh] +
             hess[it+1+it*ldh]*hess[it+1+it*ldh]);
      givens_c[it] = hess[it+it*ldh] / delta;
      givens_s[it] = hess[it+1+it*ldh] / delta;
      hess[it+it*ldh] = givens_c[it]*hess[it+it*ldh]
        + givens_s[it]*hess[it+1+it*ldh];
      b_[it+1] = -givens_s[it]*b_[it];
      b_[it] = givens_c[it]*b_[it];
      rho = abs(b_[it+1]);
      // std::cout << "GMRES it. " << totit << "\tres = "
      //           << std::setw(12) << rho
      //           << "\trel.res = " << std::setw(12)
      //           << rho/rho0 << std::endl;
      if ((rho < tol) || (rho/rho0 < tol) || (totit >= maxit)) {
        no_conv = false;
        nrit = it;
        break;
      }
    }
    for (int k=nrit; k>=0; k--) {
      for (int i=k+1; i<=nrit; i++)
        b_[k] -= hess[k+i*ldh]*b_[i];
      b_[k] /= hess[k+k*ldh];
    }
    for (int i=0; i<=nrit; i++)
      qd_axpy(b_[i], Z[i], x);
  }
  std::vector<qd_real> res(N);
  qd_residual(spmv, x, b, res);
  std::cout << "QD-FGMRES it= " << totit
            << " ||r||= " << qd_norm(res) << std::endl;
  to_Number(x, b);
}


/*
 * https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=f4883283029848a402f6200d963941ea5bba6f1c
 */
template<typename SPMV, typename PREC>
void qd_minres2(Index N, const SPMV& spmv, const PREC& prec, Number* b) {
  std::vector<qd_real> v_1(N), v(N), w_1(N), w(N), x(N), Av(N);
  v.assign(b, b+N);
  prec(v);
  int maxit = 20;
  std::vector<qd_real> beta(maxit+1), gamma(maxit+2), sigma(maxit+2);
  beta[0] = qd_norm(v);
  auto eta = beta[0];
  gamma[0] = gamma[1] = qd_real(1.);
  qd_real tol("1e-60");
  for (int it=0; it<maxit; it++) {
    for (Index i=0; i<N; i++)
      v[i] /= beta[it];
    spmv(v, Av);
    prec(Av);
    auto alpha = qd_dot(v, Av);
    for (Index i=0; i<N; i++)
      v_1[i] = Av[i] - alpha*v[i] - beta[it]*v_1[i];
    std::swap(v_1, v);
    beta[it+1] = qd_norm(v);
    auto delta = gamma[it+1]*alpha - gamma[it]*sigma[it+1]*beta[it];
    auto rho1 = sqrt(delta*delta + beta[it+1]*beta[it+1]);
    auto rho2 = sigma[it+1]*alpha + gamma[it]*gamma[it+1]*beta[it];
    auto rho3 = sigma[it]*beta[it];
    gamma[it+2] = delta / rho1;
    sigma[it+2] = beta[it] / rho1;
    for (Index i=0; i<N; i++)
      w_1[i] = (v[i] - rho3*w_1[i] - rho2*w[i]) / rho1;
    std::swap(w_1, w);
    qd_axpy(gamma[it+2]*eta, w, x);
    eta *= sigma[it+2];
    qd_residual(spmv, x, b, Av);
    auto r_nrm = qd_norm(Av);
    std::cout << "QD-MINRES it= " << it
              << " eta= " << eta
              << " gamma[it+2]*eta= " << gamma[it+2]*eta
              << " ||r||= " << r_nrm << std::endl;
    if (eta < tol) break;
  }
  to_Number(x, b);
}


/*
 * https://en.wikipedia.org/wiki/Minimal_residual_method
 */
template<typename SPMV, typename PREC>
void qd_minres(Index N, const SPMV& spmv, const PREC& prec, Number* b) {
  std::vector<qd_real> r(N), s0(N), x(N);
  r.assign(b, b+N);
  prec(r);
  auto p0 = r;
  spmv(p0, s0);
  prec(s0);
  auto p1 = p0;
  auto s1 = s0;
  qd_real tol("1e-60"), rho, b_nrm = qd_norm(b, b+N);
  int maxit = 20;
  for (int it=0; it<maxit; it++) {
    auto p2 = p1;  p1 = p0;
    auto s2 = s1;  s1 = s0;
    auto rho_old = rho;
    rho = qd_dot(s1, s1);
    auto alpha = qd_dot(r, s1) / rho;
    // if (rho < tol) {
    //   std::cout << "MINRES breakdown, rho= " << rho
    //             << " alpha= " << alpha << std::endl;
    //   break;
    // }
    qd_axpy(alpha, p1, x);
    qd_axpy(-alpha, s1, r);
    auto r_nrm_prec = sqrt(qd_dot(r, r));
    qd_residual(spmv, x, b, p0);
    auto r_nrm = qd_norm(p0);
    auto print_MINRES = [&]() {
      std::cout << " it " << it
                << " ||Mb-MAx||= " << r_nrm_prec << " ||b-Ax||= " << r_nrm
                << " ||r||/||b||= " << r_nrm / b_nrm << std::endl;
    };
    print_MINRES();
    if (r_nrm_prec < tol) {
      std::cout << "MINRES convergence!";
      print_MINRES();
      break;
    }
    if (it == maxit - 1) {
      std::cout << "MINRES maxit!";
      print_MINRES();
      break;
    }
    p0 = s1;
    spmv(s1, s0);
    prec(s0);
    auto beta1 = qd_dot(s0, s1) / qd_dot(s1, s1);
    qd_axpy(-beta1, p1, p0);
    qd_axpy(-beta1, s1, s0);
    if (it > 1) {
      auto beta2 = qd_dot(s0, s2) / qd_dot(s2, s2);
      qd_axpy(-beta2, p2, p0);
      qd_axpy(-beta2, s2, s0);
    }
  }
  to_Number(x, b);
}

void qd_getrs(Index N, Index NZ,
              Index* ridx, Index* cidx, Number* val,
              Number* b) {
  qd_real tol("1e-60");
  std::vector<qd_real> x(N), A(N*N);
  for (Index i=0; i<NZ; i++) {
    auto row = ridx[i] - 1;
    auto col = cidx[i] - 1;
    A[row+col*N] += qd_real(val[i]);
    if (row != col)
      A[col+row*N] += qd_real(val[i]);
  }
  std::vector<Index> P(N);
  std::iota(P.begin(), P.end(), 0);
  for (Index i=0; i<N; i++) {
    qd_real maxA(0.), absA;
    Index imax = i;
    for (Index k=i; k<N; k++)
      if ((absA = fabs(A[k+i*N])) > maxA) {
        maxA = absA;
        imax = k;
      }
    if (maxA < tol)
      std::cout << "QD-LU: matrix is singular" << std::endl;
    if (imax != i) {
      std::swap(P[i], P[imax]);
      for (Index k=0; k<N; k++)
        std::swap(A[i+k*N], A[imax+k*N]);
    }
    for (Index j=i+1; j<N; j++) {
      A[j+i*N] /= A[i+i*N];
      for (Index k=i+1; k<N; k++)
        A[j+k*N] -= A[j+i*N] * A[i+k*N];
    }
  }
  for (Index i=0; i<N; i++) {
    x[i] = qd_real(b[P[i]]);
    for (Index k=0; k<i; k++)
      x[i] -= A[i+k*N] * x[k];
  }
  for (Index i=N-1; i>=0; i--) {
    for (Index k=i+1; k<N; k++)
      x[i] -= A[i+k*N] * x[k];
    x[i] /= A[i+i*N];
  }
  to_Number(x, b);
}
#endif

ESymSolverStatus Ma27TSolverInterface::Backsolve(
   Index   nrhs,
   Number* rhs_vals
)
{
   DBG_START_METH("Ma27TSolverInterface::Backsolve", dbg_verbosity);
   if( HaveIpData() )
   {
      IpData().TimingStats().LinearSystemBackSolve().Start();
   }

   Index N = dim_;
   Number* W = new Number[maxfrt_];
   Index* IW1 = new Index[nsteps_];
   Index INFO[20];

   // For each right hand side, call MA27CX
   for( Index irhs = 0; irhs < nrhs; irhs++ )
   {
      if( DBG_VERBOSITY() >= 2 )
      {
         for( Index i = 0; i < dim_; i++ )
         {
            DBG_PRINT((2, "rhs[%5d] = %23.15e\n", i, rhs_vals[irhs * dim_ + i]));
         }
      }
#if defined(QD_IR)
      unsigned int oldcw;
      fpu_fix_start(&oldcw);

      auto prec = [&](std::vector<qd_real>& v) {
        std::vector<Number> tmp(v.size());
        to_Number(v, tmp.data());
        ma27c(&N, a_, &la_, iw_, &liw_, W, &maxfrt_, tmp.data(), IW1, &nsteps_, icntl_, INFO);
        v.assign(tmp.cbegin(), tmp.cend());
      };
      auto spmv = [&](const std::vector<qd_real>& x,
                      std::vector<qd_real>& y) {
        for (auto& yi : y)
          yi = qd_real(0.);
        for (Index i=0; i<NZ_orig_; i++) {
          auto row = airn_orig_[i] - 1;
          auto col = ajcn_orig_[i] - 1;
          y[row] += qd_real(a_orig_[i]) * x[col];
          if (row != col)
            y[col] += qd_real(a_orig_[i]) * x[row];
        }
      };

      Number* b = &rhs_vals[irhs * dim_];

      // qd_getrs(N, NZ_orig_, airn_orig_, ajcn_orig_, a_orig_, b);
      qd_gmres(N, spmv, prec, b);
      // qd_minres(N, spmv, prec, b);
      // qd_minres2(N, spmv, prec, b);
      // qd_refinement(N, spmv, prec, b);

      fpu_fix_end(&oldcw);
#else
      ma27c(&N, a_, &la_, iw_, &liw_, W, &maxfrt_, &rhs_vals[irhs * dim_], IW1, &nsteps_, icntl_, INFO);
#endif
      if( DBG_VERBOSITY() >= 2 )
      {
         for( Index i = 0; i < dim_; i++ )
         {
            DBG_PRINT((2, "sol[%5d] = %23.15e\n", i, rhs_vals[irhs * dim_ + i]));
         }
      }
   }
   delete[] W;
   delete[] IW1;

   if( HaveIpData() )
   {
      IpData().TimingStats().LinearSystemBackSolve().End();
   }

   return SYMSOLVER_SUCCESS;
}

Index Ma27TSolverInterface::NumberOfNegEVals() const
{
   DBG_START_METH("Ma27TSolverInterface::NumberOfNegEVals", dbg_verbosity);
   DBG_ASSERT(ProvidesInertia());
   DBG_ASSERT(initialized_);
   return negevals_;
}

bool Ma27TSolverInterface::IncreaseQuality()
{
   DBG_START_METH("Ma27TSolverInterface::IncreaseQuality", dbg_verbosity);
   if( pivtol_ == pivtolmax_ )
   {
      return false;
   }

   pivtol_changed_ = true;

   Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA,
                  "Increasing pivot tolerance for MA27 from %7.2e ", pivtol_);
   pivtol_ = Min(pivtolmax_, std::pow(pivtol_, Number(0.75)));
   Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA,
                  "to %7.2e.\n", pivtol_);
   return true;
}

} // namespace Ipopt
