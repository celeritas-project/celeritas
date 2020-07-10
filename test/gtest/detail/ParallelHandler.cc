//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ParallelHandler.cc
//---------------------------------------------------------------------------//
#include "ParallelHandler.hh"

#include "Utils.hh"
#include "base/ColorUtils.hh"
#include "comm/Communicator.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with MPI communicator
 */
ParallelHandler::ParallelHandler(const Communicator& comm) : comm_(comm) {}

//---------------------------------------------------------------------------//
/*!
 * Print useful information at the beginning of the program
 */
void ParallelHandler::OnTestProgramStart(const ::testing::UnitTest&)
{
    if (comm_.rank() == 0)
    {
        std::cout << color_code('x') << "Testing "
#if CELERITAS_USE_MPI
                  << "on " << comm_.size() << " process"
                  << (comm_.size() > 1 ? "es" : "")
#else
                  << "in serial"
#endif
                  << color_code(' ') << std::endl;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Print useful information at the end of the program
 */
void ParallelHandler::OnTestProgramEnd(const ::testing::UnitTest&) {}

//---------------------------------------------------------------------------//
/*!
 * Barrier at the beginning of each test
 */
void ParallelHandler::OnTestStart(const ::testing::TestInfo&)
{
    comm_.barrier();
}

//---------------------------------------------------------------------------//
/*!
 * Barrier at the end of each test
 */
void ParallelHandler::OnTestEnd(const ::testing::TestInfo&)
{
    std::cout << std::flush;
    comm_.barrier();
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
