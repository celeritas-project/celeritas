//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/AtomicRelaxationReader.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>

#include "celeritas/phys/AtomicNumber.hh"

#include "ImportAtomicRelaxation.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Load the EADL atomic relaxation data.
 */
class AtomicRelaxationReader
{
  public:
    //!@{
    //! \name Type aliases
    using result_type = ImportAtomicRelaxation;
    //!@}

  public:
    // Construct the reader and locate the data using the environment variable
    AtomicRelaxationReader();

    // Construct the reader from the paths to the data directory
    explicit AtomicRelaxationReader(std::string fluor_path,
                                    std::string auger_path);

    // Read the data for the given element
    result_type operator()(AtomicNumber atomic_number) const;

  private:
    // Directory containing the EADL radiative transition data
    std::string fluor_path_;
    // Directory containing the EADL non-radiative transition data
    std::string auger_path_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
