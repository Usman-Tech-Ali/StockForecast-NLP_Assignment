import React, { useState, useRef, useEffect } from 'react';
import { Clock, ChevronDown, X } from 'lucide-react';

const HorizonSearch = ({ 
  horizons, 
  selectedHorizon, 
  onHorizonChange, 
  placeholder = "Search forecast horizons..." 
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [filteredHorizons, setFilteredHorizons] = useState(horizons);
  const searchRef = useRef(null);
  const dropdownRef = useRef(null);

  // Get selected horizon details
  const selectedHorizonData = horizons.find(horizon => horizon.value === selectedHorizon);

  // Filter horizons based on search term
  useEffect(() => {
    if (searchTerm.trim() === '') {
      setFilteredHorizons(horizons);
    } else {
      const filtered = horizons.filter(horizon =>
        horizon.label.toLowerCase().includes(searchTerm.toLowerCase()) ||
        horizon.value.toLowerCase().includes(searchTerm.toLowerCase())
      );
      setFilteredHorizons(filtered);
    }
  }, [searchTerm, horizons]);

  // Handle click outside to close dropdown
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false);
        setSearchTerm('');
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Handle keyboard navigation
  const handleKeyDown = (e) => {
    if (e.key === 'Escape') {
      setIsOpen(false);
      setSearchTerm('');
      searchRef.current?.blur();
    } else if (e.key === 'Enter' && filteredHorizons.length > 0) {
      onHorizonChange(filteredHorizons[0].value);
      setIsOpen(false);
      setSearchTerm('');
    }
  };

  const handleInputClick = () => {
    setIsOpen(true);
    searchRef.current?.focus();
  };

  const handleHorizonSelect = (value) => {
    onHorizonChange(value);
    setIsOpen(false);
    setSearchTerm('');
  };

  const clearSearch = () => {
    setSearchTerm('');
    searchRef.current?.focus();
  };

  // Format display text for horizon
  const formatHorizonDisplay = (horizon) => {
    if (!horizon) return '';
    return `${horizon.label} (${horizon.value})`;
  };

  return (
    <div className="relative" ref={dropdownRef}>
      {/* Search Input */}
      <div className="relative">
        <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
          <Clock className="h-4 w-4 text-gray-400" />
        </div>
        <input
          ref={searchRef}
          type="text"
          value={isOpen ? searchTerm : (selectedHorizonData ? formatHorizonDisplay(selectedHorizonData) : '')}
          onChange={(e) => setSearchTerm(e.target.value)}
          onKeyDown={handleKeyDown}
          onClick={handleInputClick}
          placeholder={placeholder}
          className="w-full bg-white/10 border border-white/20 rounded-lg pl-10 pr-20 py-3 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
        />
        <div className="absolute inset-y-0 right-0 flex items-center pr-2 space-x-1">
          {isOpen && searchTerm && (
            <button
              onClick={clearSearch}
              className="p-1 hover:bg-white/10 rounded-full transition-colors duration-200"
              type="button"
            >
              <X className="h-4 w-4 text-gray-400 hover:text-white" />
            </button>
          )}
          <button
            onClick={() => setIsOpen(!isOpen)}
            className="p-1 hover:bg-white/10 rounded-full transition-colors duration-200"
            type="button"
          >
            <ChevronDown className={`h-4 w-4 text-gray-400 transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`} />
          </button>
        </div>
      </div>

      {/* Dropdown Results */}
      {isOpen && (
        <div className="absolute z-50 w-full mt-2 bg-white/95 backdrop-blur-sm border border-white/20 rounded-lg shadow-xl max-h-60 overflow-y-auto">
          {filteredHorizons.length > 0 ? (
            <div className="py-2">
              {filteredHorizons.map((horizon) => (
                <button
                  key={horizon.value}
                  onClick={() => handleHorizonSelect(horizon.value)}
                  className={`w-full px-4 py-3 text-left hover:bg-blue-500/10 transition-colors duration-200 ${
                    horizon.value === selectedHorizon ? 'bg-blue-500/20 text-blue-300' : 'text-gray-800'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="font-medium text-sm">
                        {horizon.label}
                      </div>
                      <div className="text-xs text-gray-600 mt-1">
                        {horizon.value} forecast period
                      </div>
                    </div>
                    <div className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded">
                      {horizon.value}
                    </div>
                  </div>
                </button>
              ))}
            </div>
          ) : (
            <div className="px-4 py-3 text-gray-500 text-sm text-center">
              No horizons found matching "{searchTerm}"
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default HorizonSearch;
