//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
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
 * reproducibility. Note that the Celeritas interfaces uses the type-safe
 * \c AtomicNumber class but we store the atomic number as an int in
 * ImportFile.
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
    explicit AllElementReader(VecElements const& els) : elements_(els)
    {
        CELER_EXPECT(!elements_.empty());
    }

    //! Load a map of data for all stored elements
    template<class ReadOneElement>
    auto operator()(ReadOneElement&& read_el) const -> decltype(auto)
    {
        using result_type = typename ReadOneElement::result_type;

        std::map<int, result_type> result_map;

        for (ImportElement const& element : elements_)
        {
            AtomicNumber z{element.atomic_number};
            CELER_ASSERT(z);
            result_map.insert({z.unchecked_get(), read_el(z)});
        }
        return result_map;
    }

  private:
    std::vector<ImportElement> const& elements_;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
