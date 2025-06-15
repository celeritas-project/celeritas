//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/StringSimplifier.cc
//---------------------------------------------------------------------------//
#include "StringSimplifier.hh"

#include <iomanip>
#include <ios>
#include <regex>
#include <sstream>

namespace celeritas
{
namespace test
{
namespace
{
using StreamManip = std::ios_base& (*)(std::ios_base&);
std::string to_str_impl(double value, int precision, StreamManip mode)
{
    CELER_EXPECT(precision >= 0);
    std::ostringstream oss;
    oss << mode << std::setprecision(precision) << value;
    return std::move(oss).str();
}

std::string to_float(double value, int precision)
{
    CELER_EXPECT(precision >= 0);
    return to_str_impl(value, precision, std::fixed);
}

std::string to_sci(double value, int precision)
{
    CELER_EXPECT(precision > 0);
    auto s = to_str_impl(value, precision - 1, std::scientific);
    CELER_ASSERT(!s.empty());
    // Change e+001 -> e1

    // If using scientific notation, simplify the exponent
    auto exp_iter = std::find_if(
        s.begin(), s.end(), [](char c) { return c == 'e' || c == 'E'; });
    CELER_ASSERT(exp_iter != s.end());
    ++exp_iter;
    CELER_ASSERT(exp_iter != s.end());
    // Start of the exponent number, likely with leading -/+
    auto num_iter = exp_iter;
    if (*num_iter == '-')
    {
        // Don't erase the -
        ++num_iter;
        ++exp_iter;
    }
    else if (*num_iter == '+')
    {
        ++num_iter;
    }
    // Search up to the last digit
    CELER_ASSERT(num_iter != s.end());
    auto end_iter
        = std::find_if(num_iter, s.end() - 1, [](char c) { return c != '0'; });
    s.erase(exp_iter, end_iter);
    return s;
}

}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Simplify.
 */
[[nodiscard]] std::string
StringSimplifier::operator()(std::string const& old) const
{
    // Regex to match floating point numbers (both standard and scientific
    // notation) with optional scientific notation and trailing f
    static std::regex const combined_regex(
        "(?:"
        R"((-?(?:\d*\.\d+|\d+\.\d*|\d+)[eE][-+]?\d+f?)\b)"  // Sci
        "|"
        R"((-?(?:\d*\.\d+f?|\d+\.\d*f?|\d+f))(?![0-9e]+))"  // Float
        "|"
        "(\033\\[[0-9;]*m)"  // ANSI
        "|"
        "(0x[0-9a-f]+)"  // Pointer
        ")");

    std::string result;
    result.reserve(old.size());
    auto current = old.cbegin();
    std::smatch match;

    // Find all matches and process them one by one
    while (std::regex_search(current, old.cend(), match, combined_regex))
    {
        // Add the text between the last match and this one
        result.append(current, match[0].first);

        if (match[1].length() > 0)
        {
            result.append(this->simplify_sci(match[1].str()));
        }
        else if (match[2].length() > 0)
        {
            result.append(this->simplify_float(match[2].str()));
        }
        else if (match[3].length() > 0)
        {
            // Omit ANSI
        }
        else if (match[4].length() > 0)
        {
            // Add placeholder pointer
            result.append("0x0");
        }

        // Update current position
        current = match[0].second;
    }

    // Add remaining text after the last match
    result.append(current, old.cend());

    return result;
}

std::string StringSimplifier::simplify_sci(std::string s) const
{
    CELER_EXPECT(!s.empty());

    // Position is start of the mantissa, or the end of the number
    auto end = s.cend() - 1;
    if (*end != 'f')
        ++end;

    auto iter = std::find(s.cbegin(), end, '.');
    if (iter != end)
    {
        // Not e.g. 1e7
        ++iter;
    }
    auto exp_iter
        = std::find_if(iter, end, [](auto c) { return c == 'e' || c == 'E'; });

    int const precision
        = std::min<int>(1 + std::distance(iter, exp_iter), precision_);

    // Format the rounded number with appropriate notation
    s = to_sci(std::stod(s), precision);

    return s;
}

std::string StringSimplifier::simplify_float(std::string s) const
{
    CELER_EXPECT(!s.empty());

    bool const is_float = (s.back() == 'f');
    if (is_float)
    {
        // Remove float appendix
        s.pop_back();
        CELER_ASSERT(!s.empty());
    }

    // Count the number of significant digits
    auto begin = s.cbegin();
    if (*begin == '-')
    {
        ++begin;
    }
    // Strip leading zeros
    begin = std::find_if(begin, s.cend(), [](char c) { return c != '0'; });
    auto dec_iter = std::find(begin, s.cend(), '.');
    int lead_precision = std::distance(begin, dec_iter);
    if (dec_iter == s.cend())
    {
        // No decimal found: either a float (1f) or a single-digit scientific
        // (1e3)
    }
    else if (dec_iter == begin)
    {
        // Leading zeros: precision based on first digit after decimal
        lead_precision = 0;
        auto iter = std::find_if(
            dec_iter + 1, s.cend(), [](char c) { return c != '0'; });
        if (iter != s.cend())
        {
            // Strip leading zeros
            dec_iter = iter;
        }
        else
        {
            ++dec_iter;
        }
    }
    else
    {
        // Decimal is between two significant numbers
        ++dec_iter;
    }
    int dec_precision = std::distance(dec_iter, s.cend());
    int precision = std::min(lead_precision + dec_precision, precision_);

    if (precision < lead_precision)
    {
        // Leading digits are too precise: write as scientific
        s = to_sci(std::stod(s), precision);
        CELER_ASSERT(!s.empty());
    }
    else
    {
        dec_precision = std::min(dec_precision, precision_ - lead_precision);
        s = to_float(std::stod(s), dec_precision);
        if (dec_precision == 0)
        {
            // Don't make it an integer except if we parsed it as a float
            CELER_ASSERT(s.find('.') == std::string::npos);
            s.push_back('.');
        }
    }

    return s;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
