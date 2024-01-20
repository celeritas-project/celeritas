//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/OutputInterfaceAdapter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>

#include "celeritas_config.h"
#include "corecel/Assert.hh"

#include "JsonPimpl.hh"
#include "OutputInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Adapter class for writing a JSON-serializable data to output.
 *
 * This class is to be used only when JSON is available and when a \c to_json
 * free function has been defined for \c T.
 */
template<class T>
class OutputInterfaceAdapter final : public OutputInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstT = std::shared_ptr<T const>;
    using SPThis = std::shared_ptr<OutputInterfaceAdapter<T>>;
    //!@}

  public:
    // DANGEROUS helper function
    static inline SPThis
    from_const_ref(Category cat, std::string label, T const& obj);

    // Construct by capturing an object
    static inline SPThis
    from_rvalue_ref(Category cat, std::string label, T&& obj);

    // Construct from category, label, and shared pointer
    inline OutputInterfaceAdapter(Category cat, std::string label, SPConstT obj);

    //! Category of data to write
    Category category() const final { return cat_; }

    //! Label of the entry inside the category.
    std::string label() const final { return label_; }

    // Write output to the given JSON object
    void output(JsonPimpl* jp) const final;

  private:
    Category cat_;
    std::string label_;
    SPConstT obj_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from a long-lived const reference.
 *
 * This is a dangerous but sometimes necessary way to make an interface
 * adapter. The object's lifespan *must be guaranteed* to exceed that of the
 * \c OutputManager.
 *
 * Example: \code
 * output_manager.insert(OutputInterfaceAdapter<Device>::from_const_ref(
 *      OutputInterface::Category::system,
 *      "device",
 *      celeritas::device()));
 * \endcode
 */
template<class T>
auto OutputInterfaceAdapter<T>::from_const_ref(Category cat,
                                               std::string label,
                                               T const& obj) -> SPThis
{
    auto null_deleter = [](T const*) {};

    return std::make_shared<OutputInterfaceAdapter<T>>(
        cat, std::move(label), std::shared_ptr<T const>(&obj, null_deleter));
}

//---------------------------------------------------------------------------//
/*!
 * Construct by capturing an object.
 */
template<class T>
auto OutputInterfaceAdapter<T>::from_rvalue_ref(Category cat,
                                                std::string label,
                                                T&& obj) -> SPThis
{
    return std::make_shared<OutputInterfaceAdapter<T>>(
        cat, std::move(label), std::make_shared<T>(std::move(obj)));
}

//---------------------------------------------------------------------------//
/*!
 * Construct from category, label, and shared pointer.
 */
template<class T>
OutputInterfaceAdapter<T>::OutputInterfaceAdapter(Category cat,
                                                  std::string label,
                                                  SPConstT obj)
    : cat_(cat), label_(std::move(label)), obj_(std::move(obj))
{
    CELER_EXPECT(cat != Category::size_);
    CELER_EXPECT(obj_);
}

//---------------------------------------------------------------------------//
/*!
 * Write output to the given JSON object.
 */
template<class T>
void OutputInterfaceAdapter<T>::output(JsonPimpl* j) const
{
    CELER_EXPECT(j);
#if CELERITAS_USE_JSON
    to_json(j->obj, *obj_);
#else
    CELER_NOT_CONFIGURED("nlohmann_json");
#endif
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
