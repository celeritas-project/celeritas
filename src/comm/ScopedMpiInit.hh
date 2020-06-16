//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file comm/ScopedMpiInit.hh
//---------------------------------------------------------------------------//
#ifndef comm_ScopedInit_hh
#define comm_ScopedInit_hh

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * RAII class for initializing and finalizing MPI.
 */
class ScopedMpiInit
{
  public:
    // Construct with argc/argv references
    ScopedMpiInit(int* argc, char*** argv);

    //! Construct with null argc/argv when those are unavailable
    ScopedMpiInit() : ScopedMpiInit(nullptr, nullptr) {}

    // Call MPI finalize on destruction
    ~ScopedMpiInit();

    // Whether MPI has been initialized
    static bool initialized();
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#endif // comm_ScopedInit_hh
