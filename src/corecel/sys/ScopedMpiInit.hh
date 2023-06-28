//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/ScopedMpiInit.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * RAII class for initializing and finalizing MPI.
 */
class ScopedMpiInit
{
  public:
    //! Status of initialization
    enum class Status
    {
        disabled = -1,  //!< Not compiled *or* disabled via environment
        uninitialized = 0,  //!< MPI_Init has not been called anywhere
        initialized = 1  //!< MPI_Init has been called somewhere
    };

    // Whether MPI has been initialized
    static Status status();

  public:
    // Construct with argc/argv references
    ScopedMpiInit(int* argc, char*** argv);

    //! Construct with null argc/argv when those are unavailable
    ScopedMpiInit() : ScopedMpiInit(nullptr, nullptr) {}

    // Call MPI finalize on destruction
    ~ScopedMpiInit();

    //!@{
    //! Prevent moving and copying
    ScopedMpiInit(ScopedMpiInit const&) = delete;
    ScopedMpiInit& operator=(ScopedMpiInit const&) = delete;
    ScopedMpiInit(ScopedMpiInit&&) = delete;
    ScopedMpiInit& operator=(ScopedMpiInit&&) = delete;
    //!@}

  private:
    static Status status_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
