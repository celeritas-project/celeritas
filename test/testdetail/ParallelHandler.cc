//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file testdetail/ParallelHandler.cc
//---------------------------------------------------------------------------//
#include "ParallelHandler.hh"

#include "corecel/io/ColorUtils.hh"
#include "corecel/sys/MpiCommunicator.hh"
#include "corecel/sys/MpiOperations.hh"

namespace celeritas
{
namespace testdetail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with MPI communicator
 */
ParallelHandler::ParallelHandler(MpiCommunicator const& comm) : comm_(comm) {}

//---------------------------------------------------------------------------//
/*!
 * Print useful information at the beginning of the program
 */
void ParallelHandler::OnTestProgramStart(::testing::UnitTest const&)
{
    if (CELERITAS_USE_MPI && comm_.rank() == 0)
    {
        std::cout << color_code('x') << "Testing "
                  << "on " << comm_.size() << " process"
                  << (comm_.size() > 1 ? "es" : "") << color_code(' ')
                  << std::endl;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Print useful information at the end of the program
 */
void ParallelHandler::OnTestProgramEnd(::testing::UnitTest const&) {}

//---------------------------------------------------------------------------//
/*!
 * Barrier at the beginning of each test
 */
void ParallelHandler::OnTestStart(::testing::TestInfo const&)
{
    barrier(comm_);
}

//---------------------------------------------------------------------------//
/*!
 * Barrier at the end of each test
 */
void ParallelHandler::OnTestEnd(::testing::TestInfo const&)
{
    std::cout << std::flush;
    barrier(comm_);
}

//---------------------------------------------------------------------------//
}  // namespace testdetail
}  // namespace celeritas
