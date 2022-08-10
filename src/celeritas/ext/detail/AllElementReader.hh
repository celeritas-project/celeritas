//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/AllElementReader.hh
//---------------------------------------------------------------------------//
#pragma once

#include <map>
#include <vector>

#include "celeritas/io/ImportElement.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Generate a map of read data for all loaded elements.
 *
 * This can be used to load EMLOW and other data into an ImportFile for
 * reproducibility.
 */
class AllElementReader
{
  public:
    //!@{
    //! \name Type aliases
    using VecElements = std::vector<ImportElement>;
    //!@}

  public:
    //! Construct from vector of imported elements
    explicit AllElementReader(const VecElements& els) : elements_(els) {}

    //! Load a map of data for all stored elements
    template<class ReadOneElement>
    auto operator()(ReadOneElement&& read_el) const -> decltype(auto)
    {
        using AtomicNumber = ImportData::AtomicNumber;
        using result_type  = typename ReadOneElement::result_type;

        std::map<AtomicNumber, result_type> result_map;

        for (const ImportElement& element : elements_)
        {
            AtomicNumber z = element.atomic_number;
            result_map.insert({z, read_el(z)});
        }
        return result_map;
    }

  private:
    const std::vector<ImportElement>& elements_;
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
